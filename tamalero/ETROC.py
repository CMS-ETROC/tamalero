"""
For ETROC control
"""
import time
import numpy as np
from tamalero.utils import load_yaml, ffs, bit_count
from tamalero.colors import red, green, yellow
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
from random import randrange

here = os.path.dirname(os.path.abspath(__file__))

class ETROC():

    def __init__(
            self,
            rb=None,
            master='lpgbt',
            i2c_adr=0x72,
            i2c_channel=0,
            elinks={0:[0]},
            verbose=False,
            strict=True,
            reset=None,
            breed='emulator',
            vref=None,
            vref_pd=False,
            vtemp = None,
            chip_id = 0,
            no_init = False,
            hard_reset = False,
            no_hard_reset_on_init = False,
            path_to_address_table = '../address_table/ETROC2_example.yaml'
    ):
        self.QINJ_delay = 504  # this is a fixed value for the default settings of ETROC2
        self.isfake = False
        self.I2C_master = rb.DAQ_LPGBT if master.lower() == 'lpgbt' else rb.SCA
        self.master = master
        self.rb = rb
        # check if connected
        self.i2c_channel = i2c_channel
        self.i2c_adr = i2c_adr
        self.vref_pin = vref
        self.vtemp = vtemp
        self.elinks = elinks
        self.reset_pin = reset
        self.breed = breed
        self.is_connected()
        if self.connected:
            self.ver = self.get_ver()
        else:
            self.ver = "X-X-X"

        self.chip_id = chip_id
        self.module_id = chip_id >> 2
        self.chip_no = chip_id & 0x3
        self.regs = load_yaml(os.path.join(here, path_to_address_table))

        self.DAC_min  = 600  # in mV
        self.DAC_max  = 1000  # in mV
        self.DAC_step = 400/2**10
        self.invalid_FC_counter = 0

        self.hot_pixels = []

        if no_init:
            return

        if hard_reset and not no_hard_reset_on_init:
            print(f"Hard resetting the ETROCs on module {self.module_id}")
            self.reset(hard=True)
            time.sleep(0.5)
        #    self.reset(hard=True)
        #    time.sleep(1)

        # NOTE: some ETROCs need to be hard reset, otherwise the I2C target does not come alive.
        # This actually solves this issue, so please don't take it out (Chesterton's Fence, anyone?)
        for i in range(2):
            if self.is_connected():
                break
            if no_hard_reset_on_init:
                if verbose:
                    print("I would have to hard reset the ETROC, but was instructed not to do so!")
                break
            if verbose:
                print("Resetting ETROC")
            self.reset(hard=True)
            time.sleep(0.05)

        if self.is_connected():
            if vref_pd:
                self.power_down_VRef()
            else:
                self.power_up_VRef()

        self.get_elink_status()
        try:
            self.default_config(no_reset=no_hard_reset_on_init)
        except TimeoutError:
            if verbose or True:
                print("Warning: ETROC default configuration failed!")
            pass

        if self.connected:
            if not self.is_good():
                print(f"ETROC {self.chip_id} is not in the expected status! {self.controllerState=}")
                #raise RuntimeError(f"ETROC {self.chip_id} is not in the expected status! {self.controllerState=}")

        if strict:
            self.consistency(verbose=verbose)


    # =========================
    # === UTILITY FUNCTIONS ===
    # =========================
    def I2C_write(self, adr=0x0, val=0x0):
        if self.isfake:
            raise NotImplementedError("I2C read not implemented for software ETROC")
        else:
            self.I2C_master.I2C_write(
                reg=adr,  # NOTE bad naming, change?
                val=val,
                master=self.i2c_channel,
                slave_addr=self.i2c_adr,
            )

    def I2C_read(self, adr=0x0):
        if self.isfake:
            raise NotImplementedError("I2C read not implemented for software ETROC")
        else:
            return self.I2C_master.I2C_read(
                reg=adr,
                master=self.i2c_channel,
                slave_addr=self.i2c_adr,
                timeout=0.1,
            )

    def get_adr(self, reg, row=0, col=0, broadcast=False):
        tmp = []
        if self.regs[reg]['stat'] == 1 and self.regs[reg]['pixel'] == 0:
            for address in self.regs[reg]['address']:
                tmp.append(address | 0x100)
        else:
            for address in self.regs[reg]['address']:
                tmp.append(address | \
                row << 5 | \
                col << 9 | \
                broadcast << 13 | \
                self.regs[reg]['stat'] << 14 | \
                self.regs[reg]['pixel'] << 15 )
        return tmp

    def wr_adr(self, adr, val):
        if self.isfake:
            #print ("writing fake")
            self.write_adr(adr, val)
        else:
            success = False
            start_time = time.time()
            while not success:
                try:
                    self.I2C_write(adr, val)
                    success = True
                except:
                    #print(f"I2C write has failed in ETROC {self.chip_id}, retrying")
                    if time.time() - start_time > 2:
                        print(f"I2C write has failed in ETROC {self.chip_id} and retries have timed out.")
                        return 0

    def rd_adr(self, adr):
        if self.isfake:
            #print ("reading fake")
            return self.read_adr(adr)
        else:
            start_time = time.time()
            while True:
                try:
                    return self.I2C_read(adr)
                except:
                    #print(f"I2C read has failed in ETROC {self.chip_id}, retrying")
                    if time.time() - start_time > 2:
                        print(f"I2C read has failed in ETROC {self.chip_id} and retries have timed out")
                        return 0

    # read & write using register name & pix num
    def wr_reg(self, reg, val, row=0, col=0, broadcast=False):
        '''
        reg - Register name
        val - value to write
        row - arbitrary value for periphery, 0..15 for in pixel
        col - arbitrary value for periphery, 0..15 for in pixel
        broadcast - True for broadcast to all pixels
        '''
        masks    = self.regs[reg]['mask']
        shifts   = list(map(ffs, masks))
        n_bits   = [0] + list(map(bit_count, masks))
        adr      = self.get_adr(reg, row=row, col=col, broadcast=broadcast)
        if val > 2**(sum(n_bits))-1:
            raise RuntimeError(f"Value {val} is larger than the number of bits of register {reg} allow ({sum(n_bits)})")
        if self.isfake:
            #print(f"writing {adr=}, {value=}")
            adr = self.get_adr(reg, row=row, col=col, broadcast=False)
            if broadcast:
                for row in range(16):
                    for col in range(16):
                        self.wr_reg(reg, val, row=row, col=col, broadcast=False)

        n_bits_total = 0
        for i, a in enumerate(adr):
            read = self.rd_adr(a)
            value = (((val >> (n_bits[i] + n_bits_total)) << shifts[i]) & masks[i]) | (read & ~masks[i])
            n_bits_total += n_bits[i]
            self.wr_adr(a, value)


    def rd_reg(self, reg, row=0, col=0, verbose=False):
        '''
        reg - Register name
        val - value to write
        row - arbitrary value for periphery, 0..15 for in pixel
        col - arbitrary value for periphery, 0..15 for in pixel
        '''
        masks    = self.regs[reg]['mask']
        shifts   = list(map(ffs, masks))
        n_bits   = [0] + list(map(bit_count, masks))
        if verbose:
            print("Found masks", masks)
            print("Found shifts", shifts)
            print("Found n_bits", n_bits)

        adr = self.get_adr(reg, row=row, col=col)
        tmp = 0
        n_bits_total = 0
        for i, a in enumerate(adr):
            if verbose: print(i, a, masks[i], shifts[i])
            read = (self.rd_adr(a) & masks[i]) >> shifts[i]
            tmp |= (read << (n_bits[i]+n_bits_total))
            n_bits_total += n_bits[i]
        return tmp

    def print_reg_doc(self, reg=None):
        if reg==None:
            for reg in self.regs:
                self.print_reg_doc(reg)
        else:
            print("{:20s}".format(reg), [hex(x) for x in self.regs[reg]['address']], "DOC:", self.regs[reg]['doc'])

    def reset_perif(self):
        for reg in self.regs:
            if self.regs[reg]['stat'] == 0 and self.regs[reg]['pixel'] == 0:
                self.wr_reg(reg, self.regs[reg]['default'])

    def reset_pixel(self):
        for reg in self.regs:
            if self.regs[reg]['stat'] == 0 and self.regs[reg]['pixel'] == 1:
                self.wr_reg(reg, self.regs[reg]['default'], broadcast=True)

    def print_perif_stat(self):
        for reg in self.regs:
            if self.regs[reg]['stat'] == 1 and self.regs[reg]['pixel'] == 0:
                ret = self.rd_reg(reg)
                print(yellow(f"Perif status reg={reg}: ret={ret}"))

    def print_pixel_stat(self, row=0, col=0):
        for reg in self.regs:
            if self.regs[reg]['stat'] == 1 and self.regs[reg]['pixel'] == 1:
                ret = self.rd_reg(reg)
                print(yellow(f"Pixel (row={row}, col={col}) status reg={reg}: ret={ret}"))

    def print_perif_conf(self, quiet=False):
        df = []
        for reg in self.regs:
            if self.regs[reg]['stat'] == 0 and self.regs[reg]['pixel'] == 0:
                ret = self.rd_reg(reg)
                exp = self.regs[reg]['default']
                if not quiet:
                    colored = green if ret == exp else red
                    print(colored(f"Perif config reg={reg}: ret={ret}, exp={exp}"))
                df.append({'register': reg, 'value': ret, 'default': exp})
        return df

    def print_pixel_conf(self, row=0, col=0, quiet=False):
        df = []
        for reg in self.regs:
            if self.regs[reg]['stat'] == 0 and self.regs[reg]['pixel'] == 1:
                ret = self.rd_reg(reg)
                exp = self.regs[reg]['default']
                if not quiet:
                    colored = green if ret == exp else red
                    print(colored(f"Pixel (row={row}, col={col}) config reg={reg}: ret={ret}, exp={exp}"))
                df.append({'register': reg, 'value': ret, 'default': exp})
        return df

    def dump_register(self):
        reg = {}
        for r in range(16):
            for c in range(16):
                for i in range(25):
                    adr = (1 << 15) | (r<<5) | (c << 9) | i
                    reg[adr] = etroc.rd_adr(adr)
        for i in range(32):
            adr = i
            reg[adr] = etroc.rd_adr(adr)

        self.reg_dump = reg


    def pixel_sanity_check(self, full=True, verbose=False, return_matrix=False):
        all_pass = True
        nmax = 16 if full else 4  # option to make this check a bit faster
        status_matrix = np.zeros((16,16))
        if self.breed in ['emulator', 'software']:
            for row in range(nmax):
                for col in range(nmax):
                    status_matrix[row][col] = 1
        else:
            for row in range(nmax):
                for col in range(nmax):
                    ret = self.rd_reg('PixelID', row=row, col=col)
                    exp = ((col << 4) | row)
                    comp = ret == exp
                    if not comp:
                        self.hot_pixels.append((row,col))
                    if verbose:
                        if comp:
                            print(green(f"Sanity check passed for row={row}, col={col}"))
                        else:
                            print(red(f"Sanity check failed for row={row}, col={col}, expected {exp} from PixelID register but got {ret}"))
                    all_pass &= comp
                    if comp:
                        status_matrix[row][col] = 1

        if return_matrix:
            return status_matrix
        else:
            return all_pass

    def deactivate_hot_pixels(self, pixels=[], hot_pixels=True, verbose=False):
        if verbose: print("Deactivating hot pixels (row, col)")
        for row, col in pixels:
            if verbose: print(row, col)
            self.wr_reg("enable_TDC", 0, row=row, col=col)
            self.wr_reg("disDataReadout", 1, row=row, col=col)
        if hot_pixels:
            # hot pixels are those that fail the pixel sanity check
            for row, col in self.hot_pixels:
                if verbose: print(row, col)
                self.wr_reg("enable_TDC", 0, row=row, col=col)
                self.wr_reg("disDataReadout", 1, row=row, col=col)


    def pixel_random_check(self, ntest=20, verbose=False):
        all_pass = True
        for i in range(ntest):
            row = randrange(16)
            col = randrange(16)
            val = randrange(256)
            self.wr_reg('PixelSanityConfig', val, row=row, col=col)
            ret = self.rd_reg('PixelSanityStat', row=row, col=col)
            comp = val == ret
            if verbose:
                if comp:
                    print(green(f"Sanity check passed for row={row}, col={col}"))
                else:
                    print(red(f"Sanity check failed for row={row}, col={col}, expected {val} from PixelSanityStat register but got {ret}"))
            all_pass &= comp
        return all_pass

    def reset(self, hard=False):
        if self.breed not in ['software', 'emulator']:
            # the emulators are not going to be reset at all
            if hard:
                if self.rb.ver < 3:
                    self.rb.SCA.set_gpio(self.reset_pin, 0)
                    time.sleep(0.05)
                    self.rb.SCA.set_gpio(self.reset_pin, 1)
                else:
                    # NOTE: I don't like this hard coded if/else.
                    # Think about a more dynamic solution
                    self.rb.DAQ_LPGBT.set_gpio(self.reset_pin, 0)
                    time.sleep(0.05)
                    self.rb.DAQ_LPGBT.set_gpio(self.reset_pin, 1)

            else:
                if self.is_connected():
                    self.wr_reg("asyResetGlobalReadout", 0)
                    time.sleep(0.1)
                    self.wr_reg("asyResetGlobalReadout", 1)
        if not self.isfake:
            self.rb.rerun_bitslip()  # NOTE this is necessary to get the links to lock again

    def reset_modules(self):
        self.reset(hard=True)
        # reset PLL and FC modules
        self.reset_PLL()
        self.reset_fast_command()
        self.reset()
        self.default_config(no_reset=True)
        self.rb.kcu.write_node("READOUT_BOARD_%s.BITSLIP_AUTO_EN"%self.rb.rb, 0x1)
        time.sleep(0.1)
        self.rb.kcu.write_node("READOUT_BOARD_%s.BITSLIP_AUTO_EN"%self.rb.rb, 0x0)

    def read_Vref(self):
        if self.rb.ver<3:
            return self.rb.SCA.read_adc(self.vref_pin)
        else:
            return self.rb.MUX64.read_adc(self.vref_pin)

    # ============================
    # === MONITORING FUNCTIONS ===
    # ============================

    def is_connected(self):
        try:
            self.connected = self.I2C_read(0x0)  # read from first register (default value 0x2C)
        except TimeoutError:
            # this comes from lpGBT read fails
            self.connected = False
        except RuntimeError:
            # this comes from SCA read fails
            self.connected = False
        return self.connected

    def get_elink_status(self, summary=False):
        if self.isfake:
            for i in self.elinks:
                self.links_locked = {i: [True for x in self.elinks[i]]}
            #self.trig_locked = True
            #self.daq_locked = True
        else:
            # NOTE this is still old schema of DAQ and TRIG
            locked = self.rb.kcu.read_node(f"READOUT_BOARD_{self.rb.rb}.ETROC_LOCKED").value()
            locked_slave = self.rb.kcu.read_node(f"READOUT_BOARD_{self.rb.rb}.ETROC_LOCKED_SLAVE").value()
            self.links_locked = {0: [((locked >> x) & 1)==1 for x in self.elinks[0]]}
            if 1 in self.elinks:
                # if any of the elinks run through the second lpGBT
                self.links_locked.update({1: [((locked_slave >> x) & 1)==1 for x in self.elinks[1]]})

            #self.trig_locked = ((locked_slave >> self.elink) & 1) == True
            #self.daq_locked = ((locked >> self.elink) & 1) == True
        if summary:
            all_good = True
            for link in self.links_locked:
                all_good &= self.links_locked[link][0]
            return all_good
        else:
            return self.links_locked

    def get_ver(self):
        try:
            ver = [hex(self.I2C_read(i))[2:] for i in [0x19,0x18,0x17]]
            return "-".join(ver)
        except:
            return "--"

    def consistency(self, verbose=False):
        if self.isfake:
            return True
        locked = self.get_elink_status()
        if locked:
            if verbose: print("Lock status before:", locked)
            self.wr_reg('disScrambler', 0x0)
            locked1 = self.get_elink_status()
            if verbose: print("Lock status after:", locked1)
            self.wr_reg('disScrambler', 0x1)
            assert locked != locked1, "Links and I2C configuration are inconsistent, please check"
            self.get_elink_status()
        else:
            if verbose:
                print("Could not check consistency because link is not ready.")
                print(f"{self.master}, {self.i2c_channel}, {self.elink}")
        return True

    def show_occupancy(self):
        # this needs to be connected to a FIFO read - FIXME
        print ('┏' + 16*'━' + '┓')
        for i in range(16):
            print ('┃'+16*'X'+'┃')
        print ('┗' + 16*'━' + '┛')

    def show_status(self):
        self.get_elink_status()
        print("┏" + 31*'━' + "┓")
        if self.isfake:
            print("┃{:^31s}┃".format("ETROC Software Emulator"))
        else:
            print("┃{:^31s}┃".format("ETROC Hardware Emulator"))
            print("┃{:^31s}┃".format(f"{self.master} channel {self.i2c_channel}"))
            print("┃{:^31s}┃".format(f"address: {hex(self.i2c_adr)}"))
        print("┃{:^31s}┃".format("version: "+self.ver))
        print("┃" + 31*" " + "┃")
        print("┃{:^31s}┃".format(f"Link: {self.elink}"))
        print ('┃' + (green('{:^31s}'.format('lpGBT 1')) if self.links_locked else red('{:^31s}'.format('DAQ'))) + '┃' )
        print ('┃' + (green('{:^31s}'.format('lpGBT 2')) if self.trig_locked else red('{:^31s}'.format('Trigger'))) + '┃' )

        print("┗" + 31*"━" + "┛")
    # =========================
    # === CONTROL FUNCTIONS ===
    # =========================

    def default_config(self, no_reset=False):
        # FIXME should use higher level functions for better readability
        if self.is_connected():
            self.reset()  # soft reset of the global readout
            self.set_singlePort('both')
            self.set_mergeTriggerData('merge')
            self.disable_Scrambler()
            # set ETROC in 320Mbps mode
            self.wr_reg('serRateLeft', 0)
            self.wr_reg('serRateRight', 0)
            # get the current number of invalid fast commands received
            self.invalid_FC_counter = self.get_invalidFCCount()

            # give some number to the ETROC
            self.wr_reg("EFuse_Prog", (self.chip_id)<<2)  # gives the correct chip ID

            # configuration as per discussion with ETROC2 developers
            self.wr_reg("onChipL1AConf", 0)  # this should be default anyway
            self.wr_reg("PLL_ENABLEPLL", 1)
            self.wr_reg("chargeInjectionDelay", 0xa)
            self.wr_reg("L1Adelay", 0x01f5, broadcast=True)  # default for LHC / Qinj
            self.wr_reg("disTrigPath", 1, broadcast=True)
            self.wr_reg("QInjEn", 0, broadcast=True)

            ## opening TOA / TOT / Cal windows
            self.wr_reg("upperTOA", 0x3ff, broadcast=True)  # this also fixes the half-chip readout with internal test data
            self.wr_reg("lowerTOA", 0, broadcast=True)
            self.wr_reg("upperTOT", 0x1ff, broadcast=True)
            self.wr_reg("lowerTOT", 0, broadcast=True)
            self.wr_reg("upperCal", 0x3ff, broadcast=True)
            self.wr_reg("lowerCal", 0, broadcast=True)

            ## Configuring the trigger stream
            self.wr_reg("disTrigPath", 1, broadcast=True)
            self.wr_reg("upperTOATrig", 0x3ff, broadcast=True)
            self.wr_reg("lowerTOATrig", 0, broadcast=True)
            self.wr_reg("upperTOTTrig", 0x1ff, broadcast=True)
            self.wr_reg("lowerTOTTrig", 0, broadcast=True)
            self.wr_reg("upperCalTrig", 0x3ff, broadcast=True)
            self.wr_reg("lowerCalTrig", 0, broadcast=True)

            self.reset()  # soft reset of the global readout, 2nd reset needed for some ETROCs
            self.reset_fast_command()
            #
            # FIXME this is where the module_reset should happen if links are not locked??
            if not no_reset:
                elink_status = self.get_elink_status()
                #print(elink_status)
                stat = True
                for elinks in elink_status:
                    #print(elink_status[elinks])
                    for elink in elink_status[elinks]:
                        if elink == False:
                            stat &= False

                if not stat:
                    print("elinks not locked, resetting PLL and FC modules")
                    self.reset_modules()

                #print(self.get_elink_status())

    def is_good(self):
        good = True
        self.controllerState = self.rd_reg("controllerState")
        good &= (self.controllerState == 11)

        return good

    def test_config(self, occupancy=5, full_chip=False):
        '''
        custom made test configuration
        '''
        if self.is_connected():
            if full_chip:
                self.enable_data_readout(broadcast=True)
                self.wr_reg("workMode", 1, broadcast=True)
                self.wr_reg("selfTestOccupancy", occupancy, broadcast=True)

            else:
                test_pixels = [
                    (0,0),
                    (7,7),
                    (7,8),
                    (8,8),
                    (8,7),
                    (0,15),
                    (15,0),
                    (15,15),
                ]
                self.disable_data_readout(broadcast=True)
                self.wr_reg("workMode", 0, broadcast=True)
                self.wr_reg("selfTestOccupancy", 0, broadcast=True)
                for row, col in test_pixels:
                    self.enable_data_readout(row=row, col=col, broadcast=False)
                    self.wr_reg("workMode", 1, row=row, col=col, broadcast=False)
                    self.wr_reg("selfTestOccupancy", occupancy, row=row, col=col, broadcast=False)

    def physics_config(self, subset=False, offset=3, L1Adelay=None, thresholds=None, powerMode='high', out_dir=None):
        '''
        subset is either False or a list of pixels, [(1,1), (1,2), ..]
        '''
        if self.is_connected():
            self.set_power_mode(powerMode)

            if L1Adelay == None:
                L1Adelay = self.QINJ_delay
            if not subset:
                self.enable_data_readout(broadcast=True)
                self.wr_reg("workMode", 0, broadcast=True)
                self.set_L1Adelay(delay=L1Adelay, broadcast=True)
                if thresholds == None :
                    self.run_threshold_scan(offset=offset, out_dir=out_dir)
                else:
                    for row in range(16):
                        for col in range(16):
                            self.wr_reg('DAC', int(thresholds[row][col]), row=row, col=col) # want to get some noise
            else:
                self.disable_data_readout(broadcast=True)
                self.wr_reg("workMode", 0, broadcast=True)
                self.set_L1Adelay(delay=L1Adelay, broadcast=True)
                for row, col in subset:
                    self.enable_data_readout(row=row, col=col, broadcast=False)
                    if thresholds == None:
                        self.auto_threshold_scan(row=row, col=col, broadcast=False, offset=offset)
                    else:
                        self.wr_reg('DAC', int(thresholds[row][col]), row=row, col=col)

    # =======================
    # === HIGH-LEVEL FUNC ===
    # =======================
    def QInj_set(self, charge, delay, L1Adelay, row=0, col=0, broadcast=True, reset=True):
        # FIXME this is a bad name, given that set_QInj also exists
        """
        High-level function to set the charge injection in the ETROC;
        requires \'charge\' (in fC) and \'delay\' (in 781 ps steps).
        Charge injection can be done at the pixel level (\'row\', \'col\') or globally (\'broadcast\'); default is global.
        By default, the charge injection module is reset upon calling (\'reset\').
        """
        self.disable_data_readout(broadcast=True)                    #disable data readout for all pixels
        self.disable_QInj(broadcast=True)                              #disable Qinj for all pixel
        self.disable_trigger_readout(broadcast=True)                   #disable trig readout for all pix

        #Now turn on the pixel of interest
        #self.apply_THCal(row=row, col=col, broadcast=broadcast)#use auto calibration
        #self.set_THoffset(V=5, row=row, col=col, broadcast=broadcast)# Offset used to add to the auto BL for real triggering
        self.enable_data_readout(row=row, col=col, broadcast=broadcast)#enable data readout
        self.enable_QInj(row=row, col=col, broadcast=broadcast)        # Enable charge injection
        self.set_L1Adelay(delay=L1Adelay,row=row, col=col, broadcast=broadcast)#Change L1A delay
        self.enable_trigger_readout(row=row, col=col, broadcast=broadcast) #enable trigger
        self.set_QInj(charge, row=row, col=col, broadcast=broadcast)   # Set charge
        self.set_chargeInjDelay(delay)                                 # Set time delay

    def QInj_unset(self, row=0, col=0, broadcast=True):
        """
        High-level function to unset the charge injection in the ETROC.
        Unset can be done at the pixel level (\'row\', \'col\') or globally (\'broadcast\'); default is global.
        """
        if broadcast:
            self.disable_QInj(broadcast=broadcast)   # Only disable charge injection for specified pixel
        else:
            self.disable_QInj(row=row, col=col, broadcast=broadcast)   # Only disable charge injection for specified pixel

    def QInj_read(self, row=0, col=0, broadcast=True):
        if broadcast:
            qinj = [[self.get_QInj(row=y, col=x) for x in range(16)] for y in range(16)]
            return qinj
        else:
            return self.get_QInj(row=row, col=col)

    def run_threshold_scan(self, offset='auto', use=True, out_dir=None):
        from tqdm import tqdm
        baseline = np.empty([16, 16])
        noise_width = np.empty([16, 16])
        print("Running threshold scan")
        with tqdm(total=256, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for pixel in range(256):
                row = pixel & 0xF
                col = (pixel & 0xF0) >> 4
                #print(pixel, row, col)
                baseline[row][col], noise_width[row][col] = self.auto_threshold_scan(row=row, col=col, broadcast=False, offset=offset, use=use)
                #print(pixel)
                pbar.update()
        self.baseline = baseline
        self.noise_width = noise_width

        if offset == 'auto':
            thresholds = baseline + noise_width
        else:
            thresholds = baseline + offset

        if out_dir is not None:
            with open(f'{out_dir}/thresholds_module_{self.module_id}_etroc_{self.chip_no}.yaml', 'w') as f:
                dump(thresholds.tolist(), f, Dumper=Dumper,)
            with open(f'{out_dir}/baseline_module_{self.module_id}_etroc_{self.chip_no}.yaml', 'w') as f:
                dump(baseline.tolist(), f, Dumper=Dumper,)
            with open(f'{out_dir}/noise_width_module_{self.module_id}_etroc_{self.chip_no}.yaml', 'w') as f:
                dump(noise_width.tolist(), f, Dumper=Dumper,)

        return baseline, noise_width

    def plot_threshold(self, outdir='../results/', noise_width=False):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        matrix = self.baseline if not noise_width else self.noise_width
        cax = ax.matshow(matrix)
        fig.colorbar(cax,ax=ax)
        for i in range(16):
            for j in range(16):
                text = ax.text(j, i, int(matrix[i,j]),
                        ha="center", va="center", color="w", fontsize="xx-small")

        if noise_width:
            fig.savefig(f'{outdir}/module_{self.module_id}_etroc_{self.chip_no}_noise_width.png')
        else:
            fig.savefig(f'{outdir}/module_{self.module_id}_etroc_{self.chip_no}_baseline.png')


    def auto_threshold_scan(self, row=0, col=0, broadcast=False, offset='auto', time_out=3, verbose=False, use=True):
        '''
        From the manual:
        1. set "Bypass" low.
        2. set "BufEn_THCal" high.
        3. set "TH_offset" a proper value.
        4. reset "Th_Cal":
        (a) set "RSTn" low.
        (b) enable clock by issuing a rising edge of ScanStart.
        (c) set "RSTn" high.
        5. launch auto threshold calibration by issuing a rising edge of ScanStart.
        '''
        # NOTE the below routine has been checked.
        assert broadcast==False, "Auto-threshold calibration with broadcast does not work in ETROC2"
        if broadcast:
            baseline = np.empty([16, 16])
            noise_width = np.empty([16, 16])
        else:
            baseline = 0
            noise_width = 0

        self.wr_reg("enable_TDC", 0, row=row, col=col, broadcast=broadcast)
        self.wr_reg("CLKEn_THCal", 1, row=row, col=col, broadcast=broadcast)
        self.wr_reg('Bypass_THCal', 0, row=row, col=col, broadcast=broadcast)
        self.wr_reg('BufEn_THCal', 1, row=row, col=col, broadcast=broadcast)
        self.wr_reg('RSTn_THCal', 0, row=row, col=col, broadcast=broadcast)
        self.wr_reg('RSTn_THCal', 1, row=row, col=col, broadcast=broadcast)
        self.wr_reg('ScanStart_THCal', 1, row=row, col=col, broadcast=broadcast)
        done = False
        start_time = time.time()
        timed_out = False
        while not done:
            done = True
            if broadcast:
                for i in range(16):
                    for j in range(16):
                        try:
                            tmp = self.rd_reg("ScanDone", row=i, col=j)
                            done &= tmp
                        except:
                            print("ScanDone read failed.")

                #if not done: print("not done")
            else:
                try:
                    done = self.rd_reg("ScanDone", row=row, col=col)
                except:
                    print("ScanDone read failed.")
                time.sleep(0.001)
                if time.time() - start_time > time_out:
                    if verbose:
                        print(f"Auto threshold scan timed out for pixel {row=}, {col=}")
                    timed_out = True
                    break
        self.wr_reg('ScanStart_THCal', 0, row=row, col=col, broadcast=broadcast)
        self.wr_reg("CLKEn_THCal", 0, row=row, col=col, broadcast=broadcast)
        self.wr_reg('BufEn_THCal', 0, row=row, col=col, broadcast=broadcast)

        if offset == 'auto':
            if broadcast:
                # Don't care about this, broken anyway
                for i in range(16):
                    for j in range(16):
                        nw = self.get_noisewidth(row=i, col=j)
                        self.wr_reg('TH_offset', nw, row=i, col=j)
                        noise_width[i][j] = nw
                        baseline[i][j] = self.get_baseline(row=i, col=j)
            else:
                noise_width = self.get_noisewidth(row=row, col=col)
                baseline = self.get_baseline(row=row, col=col)
                self.wr_reg('Bypass_THCal', 1, row=row, col=col, broadcast=broadcast)
                if use:
                    self.wr_reg('DAC', min(baseline+noise_width, 1023), row=row, col=col, broadcast=broadcast)

        else:
            #self.wr_reg('TH_offset', offset, row=row, col=col, broadcast=broadcast)
            if broadcast:
                # broken anyway
                for i in range(16):
                    for j in range(16):
                        noise_width[i][j] = self.get_noisewidth(row=i, col=j)
                        baseline[i][j] = self.get_baseline(row=i, col=j)
            else:
                noise_width = self.get_noisewidth(row=row, col=col)
                baseline = self.get_baseline(row=row, col=col)
                self.wr_reg('Bypass_THCal', 1, row=row, col=col, broadcast=broadcast)
                if use:
                    self.wr_reg('DAC', min(baseline+offset, 1023), row=row, col=col, broadcast=broadcast)

        return baseline, noise_width

    def setup_accumulator(self, row=0, col=0):
        self.wr_reg("CLKEn_THCal", 1, row=row, col=col, broadcast=False)
        self.wr_reg("BufEn_THCal", 1, row=row, col=col, broadcast=False)
        self.wr_reg("Bypass_THCal", 1, row=row, col=col, broadcast=False)

    def check_accumulator(self, DAC, row=0, col=0):
        self.wr_reg("DAC", DAC, row=row, col=col, broadcast=False)
        self.wr_reg("RSTn_THCal", 0, row=row, col=col, broadcast=False)
        self.wr_reg("RSTn_THCal", 1, row=row, col=col, broadcast=False)
        self.wr_reg("ScanStart_THCal", 1, row=row, col=col, broadcast=False)
        self.wr_reg("ScanStart_THCal", 0, row=row, col=col, broadcast=False)
        if self.rd_reg("ScanDone", row=row, col=col):
            return self.rd_reg("ACC", row=row, col=col)
        else:
            return -1

    def internal_threshold_scan(self, row=0, col=0, dac_start=0, dac_stop=1000, dac_step=1):
        from tqdm import tqdm
        self.setup_accumulator(row=row, col=col)
        N_steps  = int((dac_stop-dac_start)/dac_step)+1 # number of steps
        dac_axis = np.linspace(dac_start, dac_stop, N_steps)
        results = []
        with tqdm(total=N_steps, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for i in range(dac_start, dac_stop+1, dac_step):
                results.append(
                    self.check_accumulator(DAC=i, row=row, col=col)
                )
                pbar.update()
        run_results = np.array(results)
        return [dac_axis, run_results]

    def get_elink_for_pixel(self, row, col):
        elinks = self.elinks[0] + self.elinks[1]
        slaves = len(self.elinks[0])*[False] + len(self.elinks[1])*[True]
        if col > 7 and self.get_singlePort() == 'both':
            # NOTE: this makes the assumption that the "right" elink is always second in the ETROC config yaml file
            return elinks[0], slaves[0]
        else:
            return elinks[1], slaves[1]
            


    # ***********************
    # *** IN-PIXEL CONFIG ***
    # ***********************

    def set_power_mode(self, mode="high", row=0, col=0, broadcast=True):
        if mode == "high":
            self.wr_reg("IBSel", 0, row=row, col=col, broadcast=broadcast)  # set into high power mode (I1 in the manual)
        elif mode == "medium":
            self.wr_reg("IBSel", 2, row=row, col=col, broadcast=broadcast)  # set into medium power mode (I2 in the manual)
        elif mode == "low":
            self.wr_reg("IBSel", 5, row=row, col=col, broadcast=broadcast)  # set into low power mode (I3 in the manual)
        elif mode == "default":
            self.wr_reg("IBSel", 7, row=row, col=col, broadcast=broadcast)  # set into default power mode (I4 in the manual)
        else:
            print(f"Don't know power mode {mode}")
        return self.rd_reg("IBSel", row=row, col=col)

    def get_power_mode(self, row=0, col=0):
        res = self.rd_reg("IBSel", row=row, col=col)
        if res == 0:
            return "high"
        elif res in [1,2,4]:
            return "medium"
        elif res in [3,5,6]:
            return "low"
        elif res == 7:
            return "default"

    # (FOR ALL PIXELS) set/get load capacitance of preamp first stage
    # 0, 80, 80, or 160 fC FIXME typo? 80 appears twice in doc
    def set_Cload(self, C, row=0, col=0, broadcast=True):
        val = {0:0b00, 40:0b01, 80:0b10, 160:0b11}
        try:
            self.wr_reg('CLsel', val(C), row=row, col=col, broadcast=broadcast)
        except KeyError:
            print('Capacitance should be 0, 80, 80, or 160 fC.')

    def get_Cload(self, row=0, col=0):
        val = {0b00:0, 0b01:40, 0b10:80, 0b11:160}
        return val[self.rd_reg('CLsel', row=row, col=col)]

    # (FOR ALL PIXELS) set/get bias current
    # I1 > I2 > I3 > I4
    def set_Ibias(self, i, row=0, col=0, broadcast=True):
        val = {1:0b000, 2:0b001, 3:0b011, 4:0b111}
        try:
            self.wr_reg('IBsel', val[i], row=row, col=col, broadcast=broadcast)
        except KeyError:
            print('Select between 1 ~ 4.')

    def get_Ibias(self, row=0, col=0):
        val = {0b000:'I1', 0b001:'I2', 0b011:'I3', 0b111:'I4'}
        return val[self.rd_reg('IBSel', row=row, col=col)]

    # (FOR ALL PIXELS) set/get feedback resistance
    # 20, 10, 5.7 or 4.4 kOhm
    def set_Rfeedback(self, R, row=0, col=0, broadcast=True):
        val = {20:0b00, 10:0b00, 5.7:0b10, 4.4:0b11}
        try:
            self.wr_reg('RfSel', val(R), row=row, col=col, broadcast=broadcast)
        except KeyError:
            print('Resistance should be 20, 10, 5.7, or 4.4 kOhms')

    def get_Rfeedback(self, row=0, col=0):
        val = {0b00:20, 0b00:10, 0b10:5.7, 0b11:4.4}
        return val[self.rd_reg('RfSel', row=row, col=col)]

    # (FOR ALL PIXELS) set/get hysteresis voltage
    # Vhys1 > Vhys2 > Vhys3 > Vhys4 > Vhys5 = 0
    def set_Vhys(self, i, row=0, col=0, broadcast=True):
        val = [0b0000, 0b0001, 0b0011, 0b0111, 0b1111]
        try:
            self.wr_reg('HysSel', val(i), row=row, col=col, broadcast=broadcast)
        except IndexError:
            print('Select between 1 ~ 5.')

    def get_Vhys(self, row=0, col=0):
        val = {0b0000:1, 0b0001:2, 0b0011:3, 0b0111:4, 0b1111:5}
        return val[self.rd_reg('HysSel', row=row, col=col)]

    # (FOR ALL PIXELS) Power up/down DAC & discriminator
    def power_up_DACDiscri(self, row=0, col=0, broadcast=True):
        self.wr_reg('PD_DACDiscri', 0, row=row, col=col, broadcast=broadcast)

    def power_down_DACDiscri(self, row=0, col=0):
        self.wr_reg('PD_DACDiscri', 1, row=row, col=col)

    # (FOR ALL PIXELS) set/get injected charge
    # 1 ~ 32 fC, typical charge is 7fC
    def set_QInj(self, C, row=0, col=0, broadcast=True):
        if C > 32:
            raise Exception('Injected charge should be < 32 fC.')
        self.wr_reg('QSel', C-1, row=row, col=col, broadcast=broadcast)

    def get_QInj(self, row=0, col=0):
        return self.rd_reg('QSel', row=row, col=col)

    # (FOR ALL PIXELS) enable/disable charge injection
    def enable_QInj(self, row=0, col=0, broadcast=True):
        self.wr_reg('QInjEn', 1, row=row, col=col, broadcast=broadcast)

    def disable_QInj(self, row=0, col=0, broadcast=True):
        self.wr_reg('QInjEn', 0, row=row, col=col, broadcast=broadcast)

    # (FOR ALL PIXELS) TDC control
    # TDC automatically reset controller for every clock period
    def autoReset_TDC(self, row=0, col=0, broadcast=True):
        self.wr_reg('autoReset_TDC', 1, row=row, col=col, broadcast=broadcast)

    # enable/disable TDC conversion
    def enable_TDC(self, row=0, col=0, broadcast=True):
        self.wr_reg('enable_TDC', 1, row=row, col=col, broadcast=broadcast)

    def disable_TDC(self, row=0, col=0, broadcast=True):
        self.wr_reg('enable_TDC', 0, row=row, col=col, broadcast=broadcast)

    # Bit width of bubble tolerant in TDC encode
    def set_level_TDC(self, w, row=0, col=0, broadcast=True):
        if w > 0b011:
            raise Exception('bit width can be up to 0b011.')
        self.wr_reg('level_TDC', w, row=row, col=col, broadcast=broadcast)

    def get_level_TDC(self, row=0, col=0):
        return self.rd_reg('level_TDC', row=row, col=col)

    # Reset TDC encoder, active low
    def reset_TDC(self, row=0, col=0, broadcast=True):
        self.wr_reg('resetn_TDC', 0, row=row, col=col, broadcast=broadcast) #FIXME reg name has typo in doc?

    # enable/disable test mode where TDC generates a fixed test pulse as input signal for test for every 25 ns
    def enable_TDC_testMode(self, row=0, col=0, broadcast=True):
        self.wr_reg('testMode_TDC', 1, row=row, col=col, broadcast=broadcast)

    def disable_TDC_testMode(self, row=0, col=0, broadcast=True):
        self.wr_reg('testMode_TDC', 0, row=row, col=col, broadcast=broadcast)

    # (FOR ALL PIXELS) THCal control
    # Bypass/apply in-pixel threshold calibration block
    def bypass_THCal(self, row=0, col=0, broadcast=True):
        self.wr_reg('Bypass_THCal', 1, row=row, col=col, broadcast=broadcast)

    def apply_THCal(self, row=0, col=0, broadcast=True):
        self.wr_reg('Bypass_THCal', 0, row=row, col=col, broadcast=broadcast)

    # When Bypass_THCal = 1, TH = DAC
#    def set_Vth_pix(self, vth, row=0, col=0, broadcast=True):
#        self.wr_reg('DAC', vth, row=row, col=col, broadcast=broadcast)
#
#    def get_Vth_pix(self, row=0, col=0):
#        return self.rd_reg('DAC', row=row, col=col)
    def set_Vth_mV(self, vth, row=0, col=0, broadcast=True):
        # From the Manual:
        # DAC from 0.6-1.0V (400mV), step size 0.4mV (400mV/2**10)
        assert self.DAC_min < vth < self.DAC_max, "vth out of range: 600-1000mV"
        th = round((vth-self.DAC_min)/self.DAC_step)
        self.wr_reg('DAC', th, row=row, col=col, broadcast=broadcast)

    def get_Vth_mV(self, row=0, col=0):
        th = self.rd_reg('DAC', row=row, col=col)
        return th*self.DAC_step + self.DAC_min

    # Threshold offset for calibrated baseline. TH = BL + TH_offset
    def set_THoffset(self, V, row=0, col=0, broadcast=True):
        self.wr_reg('TH_offset', V, row=row, col=col, broadcast=broadcast)

    def get_THoffset(self, row=0, col=0):
        return self.rd_reg('TH_offset', row=row, col=col)

    def add_THoffset(self, val, row=0, col=0, broadcast=False):
        '''
        add to the currently set offset
        '''
        rows = range(16) if broadcast else [row]
        cols = range(16) if broadcast else [col]
        for i in rows:
            for j in cols:
                tmp = self.get_THoffset(row=i, col=j)
                self.set_THoffset(tmp+val, row=i, col=j, broadcast=False)

    # Reset of threshold calibration block, active low
    def reset_THCal(self, row=0, col=0, broadcast=True):
        self.wr_reg('RSTn_THCal', 0, row=row, col=col, broadcast=broadcast)

    # Initialize threshold calibration
    def init_THCal(self, row=0, col=0, broadcast=True): #FIXME better name?
        self.wr_reg('ScanStart_THCal', 1, row=row, col=col, broadcast=broadcast)

    # Enable/disable threshold calibration buffer
    def enable_THCal_buffer(self, row=0, col=0, broadcast=True):
        self.wr_reg('BufEn_THCal', 1, row=row, col=col, broadcast=broadcast)

    def disable_THCal_buffer(self, row=0, col=0, broadcast=True):
        self.wr_reg('BufEn_THCal', 0, row=row, col=col, broadcast=broadcast)

    # Enable/disable threshold calibration clock. Only used when threshold calibration clock is bypassed.
    def enable_THCal_clock(self, row=0, col=0, broadcast=True):
        self.wr_reg('CLKEn_THCal', 1, row=row, col=col, broadcast=broadcast)

    def disable_THCal_clock(self, row=0, col=0, broadcast=True):
        self.wr_reg('CLKEn_THCal', 0, row=row, col=col, broadcast=broadcast)

    # (FOR ALL PIXELS) Readout control
    # Readout work mode selection
    def set_workMode(self, mode, row=0, col=0, broadcast=True):
        val = {'normal': 0b00, 'self test fixed': 0b01, 'self test random': 0b10}
        try:
            self.wr_reg('workMode', val[mode], row=row, col=col, broadcast=broadcast)
        except KeyError:
            print('Choose between \'normal\', \'self test fixed\', \'self test random\'.')

    def get_workMode(self, row=0, col=0):
        val = {0b00:'normal', 0b01:'self test fixed', 0b10:'self test random'}
        return val[self.wr_reg('workMode', row=row, col=col)]

    # L1A latency
    def set_L1Adelay(self, delay, row=0, col=0, broadcast=True):
        self.wr_reg('L1Adelay', delay, row=row, col=col, broadcast=broadcast)

    def get_L1Adelay(self, row=0, col=0):
        return self.rd_reg('L1Adelay', row=row, col=col)

    # Enable/disable TDC data readout of current pixel
    def enable_data_readout(self, row=0, col=0, broadcast=True):
        self.wr_reg('disDataReadout', 0, row=row, col=col, broadcast=broadcast)

    def disable_data_readout(self, row=0, col=0, broadcast=True):
        self.wr_reg('disDataReadout', 1, row=row, col=col, broadcast=broadcast)

    # Enable/disable trger readout of current pixel
    def enable_trigger_readout(self, row=0, col=0, broadcast=True):
        self.wr_reg('disTrigPath', 0, row=row, col=col, broadcast=broadcast)

    def disable_trigger_readout(self, row=0, col=0, broadcast=True):
        self.wr_reg('disTrigPath', 1, row=row, col=col, broadcast=broadcast)

    # Set upper/lower thresholds for trigger readout of TOA, TOT, Cal
    def set_trigger_TH(self, datatype, upper=None, lower=None, row=0, col=0, broadcast=True):
        if datatype not in ['TOA', 'TOT', 'Cal']:
            raise Exception('type of data should be TOA, TOT or CAL.')
        if upper is not None:
            self.wr_reg('upper'+datatype+'Trig', upper, row=row, col=col, broadcast=broadcast)
        if lower is not None:
            self.wr_reg('lower'+datatype+'Trig', lower, row=row, col=col, broadcast=broadcast)

    def get_trigger_TH(self, datatype, row=0, col=0):
        if datatype not in ['TOA', 'TOT', 'Cal']:
            raise Exception('type of data should be TOA, TOT or CAL.')
        upper = 'upper'+datatype+'Trig'
        lower = 'lower'+datatype+'Trig'
        return self.rd_reg(upper, row=row, col=col), self.rd_reg(lower, row=row, col=col)

    # Set upper/lower thresholds for TDC data readout of TOA, TOT, Cal
    def set_data_TH(self, datatype, upper=None, lower=None, row=0, col=0, broadcast=True):
        if datatype not in ['TOA', 'TOT', 'Cal']:
            raise Exception('type of data should be TOA, TOT or Cal.')
        if upper is not None:
            self.wr_reg('upper'+datatype, upper, row=row, col=col, broadcast=broadcast)
        if lower is not None:
            self.wr_reg('lower'+datatype, lower, row=row, col=col, broadcast=broadcast)

    def get_data_TH(self, datatype, row=0, col=0):
        if datatype not in ['TOA', 'TOT', 'Cal']:
            raise Exception('type of data should be TOA, TOT or Cal.')
        upper = 'upper'+datatype
        lower = 'lower'+datatype
        return self.rd_reg(upper, row=row, col=col), self.rd_reg(lower, row=row, col=col)

    # Enable/disable circular buffer write address offset
    def enable_adr_offset(self, row=0, col=0, broadcast=True):
        self.wr_reg('addrOffset', 1, row=row, col=col, broadcast=broadcast)

    def disable_adr_offset(self, row=0, col=0, broadcast=True):
        self.wr_reg('addrOffset', 0, row=row, col=col, broadcast=broadcast)

    # Self-test data occupancy is selfTestOccupancy[6:0]/128
    def set_selftest_occupancy(self, occ, row=0, col=0, broadcast=True):
        self.wr_reg('selfTestOccupancy', occ, row=row, col=col, broadcast=broadcast)

    def get_selftest_occupancy(self, row=0, col=0):
        return self.rd_reg('selfTestOccupancy', row=row, col=col)


    # ***********************
    # *** IN-PIXEL STATUS ***
    # ***********************

    # (FOR ALL PIXELS) Accumulator of the threshold calibration
    def get_ACC(self, row=0, col=0):
        return self.rd_reg('ACC', row=row, col=col)

    # (FOR ALL PIXELS) Scan done signal of the threshold calibration
    def is_scanDone(self, row=0, col=0):
        result = self.rd_reg('ScanDone', row=row, col=col)
        if result == 1:
            return True
        else:
            return False

    # (FOR ALL PIXELS) Baseline obtained from threshold calibration
    def get_baseline(self, row=0, col=0):
        return self.rd_reg('BL', row=row, col=col)

    # (FOR ALL PIXELS) Noise width from threshold calibration. Expect less than 10.
    def get_noisewidth(self, row=0, col=0):
        return self.rd_reg('NW', row=row, col=col)

    # (FOR ALL PIXELS) 10-bit threshold applied to the DAC input
    def get_threshold(self, row=0, col=0):
        return self.rd_reg('TH', row=row, col=col)

    # (FOR ALL PIXELS) Threshold calibration state machine output
    def get_THstate(self, row=0, col=0):
        return self.rd_reg('THstate', row=row, col=col)

    # (FOR ALL PIXELS) Col[3:0], Row[3:0]
    def get_pixelID(self, row=0, col=0):
        return self.rd_reg('PixelID', row=row, col=col)

    # ***********************
    # **** PERIPH CONFIG ****
    # ***********************

    # Phase delay of readout clock, 780 ps a step (Pixel or Global)
    def set_readoutClkDelay(self, clk, delay):
        if clk not in ['Pixel', 'Global']:
            raise Exception('Clock should be either Pixel or Global')
        self.wr_reg('readoutClockDelay'+clk, delay)

    def get_readoutClkDelay(self, clk):
        if clk not in ['Pixel', 'Global']:
            raise Exception('Clock should be either Pixel or Global')
        return self.rd_reg('readoutClockDelay'+clk)

    # Positive pulse width of readout clock, 780 ps a step (Pixel or Global)
    def set_readoutClkWidth(self, clk, width):
        if clk not in ['Pixel', 'Global']:
            raise Exception('Clock should be either Pixel or Global')
        self.wr_reg('readoutClockWidth'+clk, width)

    def get_readoutClkWidth(self, clk):
        if clk not in ['Pixel', 'Global']:
            raise Exception('Clock should be either Pixel or Global')
        return self.rd_reg('readoutClockWidth'+clk)

    # Data rate selection of Right or Left serial port
    def set_dataRate(self, port, rate):
        if port not in ['Left', 'Right']:
            raise Exception('Choose between Left or Right serial port')
        val = {320:0b00, 640:0b01, 1280:0b10}
        try:
            self.wr_reg('serRate'+port, val[rate])
        except KeyError:
            print('Choose between rates of 320 Mbps, 640 Mbps and 1280 Mbps')

    def get_dataRate(self, port):
        if port not in ['Left', 'Right']:
            raise Exception('Choose between Left or Right serial port')
        val = {0b00:320, 0b01:640, 0b10:1280}
        return val[self.rd_reg('serRate'+port)]

    # Link reset test pattern selection
    def set_linkResetTestPattern(self, mode):
        val = {'PRBS':0b0, 'Fixed pattern':0b1}
        try:
            self.wr_reg('linkResetTestPattern', val[mode])
        except KeyError:
            print('Choose between \'PRBS\' and \'Fixed pattern\' selections')

    def get_linkResetTestPattern(self):
        val = {0b0:'PRBS', 0b1:'Fixed pattern'}
        return val[self.rd_reg('linkResetTestPattern')]

    # User-specified pattern to be sent during link reset, LSB first
    def set_linkResetFixedPattern(self, pattern):
        self.wr_reg('linkResetFixedPattern', pattern)

    def get_linkResetFixedPattern(self):
        return self.rd_reg('linkResetFixedPattern')

    # Empty BCID slot for synchronization
    def set_BCID(self, bcid):
        self.wr_reg('emptySlotBCID', bcid)

    def get_BCID(self):
        return self.rd_reg('emptySlotBCID')

    # Trigger data size, can be 0, 1, 2, 4, 8, 16
    def set_triggerGranularity(self, size):
        val = {0:0, 1:1, 2:2, 4:3, 8:4, 16:5}
        try:
            self.wr_reg('triggerGranularity', val[size])
        except KeyError:
            print('Trigger data size can only be 0, 1, 2, 4, 8 or 16')

    def get_triggerGranularity(self):
        val = {0:0, 1:1, 2:2, 3:4, 4:8, 5:16, 6:0, 7:0}
        return val[self.rd_reg('triggerGranularity')]

    # Enable/disable scrambler
    def enable_Scrambler(self):
        self.wr_reg('disScrambler', 0)

    def disable_Scrambler(self):
        self.wr_reg('disScrambler', 1)

    # Merge trigger and data in a port
    def set_mergeTriggerData(self, mode):
        val = {'separate':0, 'merge':1}
        if (self.get_singlePort() == 'both') and (mode == 'separate'):
            raise Exception('Trigger and data in different ports is only allowed when singlePort is set to \'right\'')
        try:
            self.wr_reg('mergeTriggerData', val[mode])
        except KeyError:
            print('Choose between \'merge\' and \'separate\' options')

    def get_mergeTriggerData(self):
        val = {0: 'separate', 1:'merge'}
        return val[self.rd_reg('mergeTriggerData')]

    # Enable single port (right) or both ports
    def set_singlePort(self, mode):
        val = {'both':0, 'right':1}
        try:
            self.wr_reg('singlePort', val[mode])
        except KeyError:
            print('Choose between \'both\' and \'right\' options')

    def get_singlePort(self):
        val = {0:'both', 1:'right'}
        return val[self.rd_reg('singlePort')]

    # On-chip L1A mode
    def set_l1aMode(self, mode):
        val = {'disable':0b00, 'periodic':0b10, 'random':0b11}
        try:
            self.wr_reg('onChipL1AConf', val[mode])
        except KeyError:
            print('Choose between \'disable\', \'periodic\' and \'random\' options')

    def get_l1aMode(self):
        val = {0b00:'disable', 0b10:'periodic', 0b11:'random', 0b01:'disable'}
        return val[self.rd_reg('onChipL1AConf')]

    # BCID when BCID is reset
    def set_BCIDoffset(self, offset):
        self.wr_reg('BCIDoffset', offset)

    def get_BCIDoffset(self, offset):
        self.rd_reg('BCIDoffset')

    # Fast command decoder self-alignment or manual alignment
    def set_fcAlign(self, mode):
        val = {'manual':0, 'self':1}
        try:
            self.wr_reg('fcSelfAlignEn', val[mode])
        except KeyError:
            print('Choose between \'manual\' and \'self\' options')

    def get_fcAlign(self):
        val = {0:'manual', 1:'self'}
        return val[self.rd_reg('fcSelfAlignEn')]

    # Enable/disable clock delay in fast command manual alignment mode
    def enable_fcClkDelay(self):
        assert self.get_fcAlign() == 'manual'
        self.wr_reg('fcClkDelayEn', 1)

    def disable_fcClkDelay(self):
        assert self.get_fcAlign() == 'manual'
        self.wr_reg('fcClkDelayEn', 0)

    # Enable/disable data delay in fast command manual alignment mode, active high
    def enable_fcDataDelay(self):
        assert self.get_fcAlign() == 'manual'
        self.wr_reg('fcDataDelayEn', 1)

    def disable_fcDataDelay(self):
        assert self.get_fcAlign() == 'manual'
        self.wr_reg('fcDataDelayEn', 0)

    # The charge injection delay to the 40 MHz clock rising edge. Start from rising edge
    # of 40 MHz clock, each step 781 ps. The pulse width is fixed of 50 ns.
    def set_chargeInjDelay(self, delay):
        self.wr_reg('chargeInjectionDelay', delay)

    def get_chargeInjDelay(self):
        return self.rd_reg('chargeInjectionDelay')

    # TDC Reference strobe selection
    def set_refStr(self, refStr):
        self.wr_reg('RefStrSel', refStr)

    def get_refStr(self):
        return self.rd_reg('RefStrSel')

    # Charge pump bias current selection, [0:8:120] uA. Debugging use only.
    def set_PLLBiasGen(self, bias):
        self.wr_reg('PLL_BIASGEN_CONFIG', bias)

    def get_PLLBiasGen(self):
        return self.rd_reg('PLL_BIASGEN_CONFIG')

    # Bias current selection of the I-filter (0:1.1:8 uA) or P-filter (0:5.46:82 uA) unit cell in PLL mod. Debugging use only.
    def set_PLLConfig(self, filt, bias):
        if filt not in ['I', 'P']:
            raise Exception('Choose between \'I\' or \'P\' filter')
        self.wr_reg('PLL_CONFIG_'+filt+'_PLL', bias)

    def get_PLLConfig(self, filt):
        if filt not in ['I', 'P']:
            raise Exception('Choose between \'I\' or \'P\' filter')
        return self.rd_reg('PLL_CONFIG_'+filt+'_PLL')

    # Resistor selection of the P-path in PLL mode [R=1/2*79.8k/CONFIG] Ohm. Debugging use only.
    def set_PLLRes(self, R):
        config = 1 / (2 * 79.8e3) / R
        self.wr_reg('PLL_R_CONFIG', config)

    def get_PLLRes(self):
        config = self.rd_reg('PLL_R_CONFIG')
        R = 1 / (2 * 79.8e3) / config
        return R

    # Bias current selection of the VCO core [0:0.470:7.1] mA. Debugging use only.
    def set_PLLvco(self, bias):
        self.wr_reg('PLL_vcoDAC', bias)

    def get_PLLvco(self):
        return self.rd_reg('PLL_vcoDAC')

    # Output rail-to-rail mode selection of the VCO, active low. Debugging use only.
    def set_PLLvcoRail(self, mode):
        if mode not in ['rail', 'CML']:
            raise Exception('Chose between \'rail\' and \'CML\' options')
        val = {'rail':0, 'CML':1}
        self.wr_reg('PLL_vcoRailMode', val[mode])

    def get_PLLvcoRail(self):
        val = {0:'rail', 'CML':1}
        return val[self.rd_reg('PLL_vcoRailMode')]

    # Enable/disable PLL mode, active high. Debugging use only.
    def enable_PLL(self):
        self.wr_reg('PLL_ENABLEPLL', 1)

    def disable_PLL(self):
        self.wr_reg('PLL_ENABLEPLL', 0)

    # Adjusting the phase of the output clk1G28 of freqPrescaler in the feedback
    # divider (N=64) by one skip from low to high. Debugging use only.
    def set_PLLFBDiv(self, skip):
        self.wr_reg('PLL_FBDiv_skip', skip)

    def get_PLLFBDiv(self):
        return self.rd_reg('PLL_FBDiv_skip')

    # Enable/disable feedback divider
    def enable_PLLFB(self):
        self.wr_reg('PLL_FBDiv_clkTreeDisable', 0)

    def disable_PLLFB(self):
        self.wr_reg('PLL_FBDiv_clkTreeDisable', 1)

    # Enable/disable output clocks for serializer
    def enable_PLLclkSer(self):
        self.wr_reg('PLLclkgen_disSER', 0)

    def disable_PLLclkSer(self):
        self.wr_reg('PLLclkgen_disSER', 1)

    # Enable/disable VCO output buffer (associated with clk5g12lshp, clk5g12lshn), active high.
    # clk5g12lsh is the output clock of the first input buffer in prescaler, and the source
    # clock for all output clocks. Once disabled, all output clocks are disabled. Debugging use only.
    def enable_PLLvcoBuff(self):
        self.wr_reg('PLLclkgen_disVCO', 0)

    def disable_PLLvcoBuff(self):
        self.wr_reg('PLLclkgen_disVCO', 1)

    # Enable/disable output clocks for EOM, active high. When PLLclkgen_disEOM is high, the following
    # clocks are disabled: clk5g12EOMp, clk5g12EOMn. Debugging use only.
    def enable_PLLEOM(self):
        self.wr_reg('PLLclkgen_disEOM', 0)

    def disable_PLLEOM(self):
        self.wr_reg('PLLclkgen_disEOM', 1)

    # Enable/disable the internal clock buffers and 1/2 clock divider in prescaler, active high. When
    # PLLclkgen_disCLK is high, all output clocks are disabled. Debugging use only.
    def enable_PLLclk(self):
        self.wr_reg('PLLclkgen_disCLK', 0)

    def disable_PLLclk(self):
        self.wr_reg('PLLclkgen_disCLK', 1)

    # Enable/disable output clocks for deserializer, active high. When PLLclkgen_disDES is high, the
    # following clocks are disabled: clk2g56Qp, clk2g56Qn, clk2g56lp, clk2g56ln. clk2g56Q is
    # the 2.56 GHz clock for test in ETROC_PLL. clk2g56Q is used as WS clock in ETROC2. Debugging use only.
    def enable_PLLDes(self):
        self.wr_reg('PLLclkgen_disDES', 0)

    def disable_PLLDes(self):
        self.wr_reg('PLLclkgen_disDES', 1)

    # Selecting PLL clock or off-chip clock for TDC and readout. Debugging use only.
    def set_CLKSel(self, clk):
        if clk not in ['off-chip', 'PLL']:
            raise Exception('Choose between \'off-chip\' and \'PLL\' options')
        val = {'off-chip':0, 'PLL':1}
        self.wr_reg('CLKSel', val[clk])

    def get_CLKSel(self):
        val = {0:'off-chip', 1:'PLL'}
        return val[self.rd_reg('CLKSel')]

    # Charge pump current control bits, range from 0 to 15uA for charge and discharge. Debugging use only.
    def set_CPCurrent(self, current):
        self.wr_reg('PS_CPCurrent', current)

    def get_CPCurrent(self):
        return self.rd_reg('PS_CPCurrent')

    # Reset the control voltage of DLL to power supply, active high. Debugging use only.
    def toggle_CapRst(self):
        val = ~self.get_CapRst()
        self.wr_reg('PS_CapRst', val)

    def get_CapRst(self):
        return self.rd_reg('PS_CapRst')

    # Enable/disable DLL, active high. Debugging use only.
    def enable_DLL(self):
        self.wr_reg('PS_Enable', 1)

    def disable_DLL(self):
        self.wr_reg('PS_Enable', 0)

    # Force to pull down the output of the phase detector, active high. Debugging use only.
    def set_PSForceDown(self, boolean):
        if not isinstance(boolean, bool):
            raise TypeError('Argument must be a boolean')
        val = 1 if boolean else 0
        self.wr_reg('PS_ForceDown', val)

    def get_PSForceDown(self):
        val = self.rd_reg('PS_ForceDown')
        return True if val == 1 else False

    # Phase selecting control bits, PS_PhaseAdj[7:3] for coarse, PS_PhaseAdj[2:0] for fine.
    def set_PhaseAdj(self, phase):
        self.wr_reg('PS_PhaseAdj', phase)

    def get_PhaseAdj(self):
        return self.rd_reg('PS_PhaseAdj')

    # Enable/disable the Rx for the 40 MHz, 1.28 GHz reference clock or the fast command, active high. Debugging use only
    def set_Rx(self, mode, boolean):
        if not isinstance(boolean, bool):
            raise TypeError('Argument must be True (enable) or False (disable)')
        val  = 1 if boolean else 0
        regs = {40:'CLK40_EnRx', 1280:'CLK1280_EnRx', 'FC':'FC_EnRx'}
        try:
            self.wr_reg(regs[mode], val)
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    def get_Rx(self, mode):
        regs = {40:'CLK40_EnRx', 1280:'CLK1280_EnRx', 'FC':'FC_EnRx'}
        try:
            return self.rd_reg(regs[mode])
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    # Enable/disable internal termination of the Rx for the 40 MHz, 1.28 GHz reference clock or the fast command, active high. Debugging use only.
    def set_Ter(self, mode, boolean):
        if not isinstance(boolean, bool):
            raise TypeError('Argument must be True (enable) or False (disable)')
        val  = 1 if boolean else 0
        regs = {40:'CLK40_EnTer', 1280:'CLK1280_EnTer', 'FC':'FC_EnTer'}
        try:
            self.wr_reg(regs[mode], val)
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    def get_Ter(self, mode):
        regs = {40:'CLK40_EnTer', 1280:'CLK1280_EnTer', 'FC':'FC_EnTer'}
        try:
            return self.rd_reg(regs[mode])
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    # Equalization strength of the Rx for the 40 MHz, 1.28 GHz reference clock or the fast command. Debugging use only.
    # 2'b00: equalization is turned off; 2'b11: maximal equalization.
    def set_Equ(self, mode, equalization):
        regs = {40:'CLK40_Equ', 1280:'CLK1280_Equ', 'FC':'FC_Equ'}
        try:
            self.wr_reg(regs[mode], equalization)
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options for \'mode\' arg.')

    def get_Equ(self, mode):
        regs = {40:'CLK40_Equ', 1280:'CLK1280_Equ', 'FC':'FC_Equ'}
        try:
            return self.rd_reg(regs[mode])
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    # Inverting data of the Rx for the 40 MHz, 1.28 GHz reference clock or the fast command, active high. Debugging use only.
    def set_Inv(self, mode, boolean):
        if not isinstance(boolean, bool):
            raise TypeError('Argument must be True (invert) or False (don\'t invert)')
        val  = 1 if boolean else 0
        regs = {40:'CLK40_InvData', 1280:'CLK1280_InvData', 'FC':'FC_InvData'}
        try:
            self.wr_reg(regs[mode], val)
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    def get_Inv(self, mode):
        regs = {40:'CLK40_InvData', 1280:'CLK1280_InvData', 'FC':'FC_InvData'}
        try:
            return self.rd_reg(regs[mode])
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    # Set common voltage of the Rx for the 40 MHz, 1.28 GHz reference clock or the fast command to 1/2 vdd, active high. Debugging use only.
    def set_commonV(self, mode, boolean):
        if not isinstance(boolean, bool):
            raise TypeError('Argument must be True (set) or False (don\'t set)')
        val  = 1 if boolean else 0
        regs = {40:'CLK40_SetCM', 1280:'CLK1280_SetCM', 'FC':'FC_SetCM'}
        try:
            self.wr_reg(regs[mode], val)
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    def get_commonV(self, mode):
        regs = {40:'CLK40_SetCM', 1280:'CLK1280_SetCM', 'FC':'FC_SetCM'}
        try:
            return self.rd_reg(regs[mode])
        except KeyError:
            print('Choose between 40 (40 MHz ref. clock), 1280 (1.28 GHz ref. clock) and \'FC\' (fast command) options')

    # Enable/disable the power up sequence, active high
    def enable_PowerUp(self):
        self.wr_reg('disPowerSequence', 0)

    def disable_PowerUp(self):
        self.wr_reg('disPowerSequence', 1)

    # Reset power sequencer controller, active high
    def reset_Power(self):
        self.wr_reg('softBoot', 1)

    # The register controlling the SCLK pulse width, ranging ranges from 3 us to 10 us with step of 0.5 us.
    # The default value is 4 corresponding to 5 us pulse width. Debugging use only.
    def set_SCLKWidth(self, width):
        self.wr_reg('EFuse_TCKHP', width)

    def get_SCLKWidth(self):
        return self.rd_reg('EFuse_TCKHP')

    # Enable/disable EFuse clock
    def enable_EFuseClk(self):
        self.wr_reg('EFuse_EnClk', 1)

    def disable_EFuseClk(self):
        self.wr_reg('EFuse_EnClk', 0)

    # Operation mode of EFuse.
    # 2'b01: programming mode;
    # 2'b10: reading mode.
    def set_EFuseMode(self, mode):
        val = {'programming':0b01, 'reading':0b10}
        try:
            self.wr_reg('EFuse_Mode', val[mode])
        except KeyError:
            print('Choose between \'programming\' and \'reading\' options')

    def get_EFuseMode(self):
        val = {0b01:'programming', 0b10:'reading'}
        return val[self.rd_reg('EFuse_Mode')]

    # Reset signal of the EFuse controller, active low
    def reset_EFuse(self):
        self.wr_reg('EFuse_Rstn', 0)

    # Start signal of the EFuse programming. A positive pulse will start the programming
    def start_EFuse(self):
        self.wr_reg('EFuse_Start', 1)

    # Data to be written into EFuse
    def set_EFuseDat(self, data):
        self.wr_reg('EFuse_Prog', data)

    def get_EFuseDat(self):
        return self.rd_reg('EFuse_Prog')

    # Bypass EFuse.
    # 1'b0: EFuse output Q[31:0] is output;
    # 1'b1: EFuse raw data from I2C (EFuse_Prog[31:0]) is output
    def bypass_EFuse(self, bypass):
        if not isinstance(bypass, bool):
            raise TypeError('Argument must be True (bypass) or False (don\'t bypass)')
        val = 1 if bypass else 0
        self.wr_reg('EFuse_Bypass', val)

    # If the number of instantLock is true for 2^IfLockThrCounter in a row, the PLL is locked in the initial status
    def set_IfLockThrCounter(self, counter):
        self.wr_reg('IfLockThrCounter', counter)

    def get_IfLockThrCounter(self):
        return self.rd_reg('IfLockThrCounter')

    # If the number of instantLock is true for 2^IfReLockThrCounter in a row, the PLL is relocked before the unlock status is confirmed
    def set_IfReLockThrCounter(self, counter):
        self.wr_reg('IfReLockThrCounter', counter)

    def get_IfReLockThrCounter(self):
        return self.rd_reg('IfReLockThrCounter')

    # If the number of instantLock is false for 2^IfUnLockThrCounter in a row, the PLL is unlocked
    def set_IfUnLockThrCounter(self, counter):
        self.wr_reg('IfUnLockThrCounter', counter)

    def get_IfUnLockThrCounter(self):
        return self.rd_reg('IfUnLockThrCounter')

    # The fast command bit clock alignment command is issued by I2C.
    # Used in self-alignment only.
    # Initializing the clock phase alignment process at its rising edge (synchronized by the 40 MHz PLL clock)
    def enable_FCClkPhaseAlign(self):
        assert self.get_fcAlign() == 'self'
        self.wr_reg('asyAlignFastcommand', 1)

    def disable_FCClkPhaseAlign(self):
        assert self.get_fcAlign() == 'self'
        self.wr_reg('asyAlignFastcommand', 0)

    # Link reset signal from I2C, active high. If it is high, ETROC2 sends test pattern via link
    def set_LinkReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 1 if reset else 0
        self.wr_reg('asyLinkReset', val)

    def get_LinkReset(self):
        return self.rd_reg('asyLinkReset')

    # Reset PLL AFC from I2C, active low
    def set_PLLReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 0 if reset else 1
        self.wr_reg('asyPLLReset', val)

    def get_PLLReset(self):
        return self.rd_reg('asyPLLReset')

    def reset_PLL(self):
        self.wr_reg("asyPLLReset", 0)
        time.sleep(0.1)
        self.wr_reg("asyPLLReset", 1)

    # Reset charge injection module, active low
    def reset_charge_injection(self):
        self.wr_reg("asyResetChargeInj", 0)
        time.sleep(0.1)
        self.wr_reg("asyResetChargeInj", 1)

    # Reset fastcommand from I2C, active low
    def reset_fast_command(self):
        self.wr_reg("asyResetFastcommand", 0)
        time.sleep(0.1)
        self.wr_reg("asyResetFastcommand", 1)

    # Reset globalReadout module, active low
    def set_GlobalReadoutReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 0 if reset else 1
        self.wr_reg('asyResetGlobalReadout', val)

    def get_GlobalReadoutReset(self):
        return self.rd_reg('asyResetGlobalReadout')

    # Reset lock detect, active low (original lockDetect reset is active high, polarity changed)
    def set_LockDetectReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 0 if reset else 1
        self.wr_reg('asyResetLockDetect', val)

    def get_LockDetectReset(self):
        return self.rd_reg('asyResetLockDetect')

    # Start PLL calibration process, active high
    def start_PLLCal(self):
        self.wr_reg('asyStartCalibration', 1)

    def stop_PLLCal(self):
        self.wr_reg('asyStartCalibration', 0)

    # Power down voltage reference generator, active high.
    # 1'b1: the voltage reference generator is down.
    # 1'b0: the voltage reference generator is up.
    def power_up_VRef(self):
        self.wr_reg('VRefGen_PD', 0)

    def power_down_VRef(self):
        self.wr_reg('VRefGen_PD', 1)

    # Power down the temperature sensor, active high.
    # 1'b1: the temperature sensor is down;
    # 1'b0: the temperature sensor is up.
    def power_up_TempSen(self):
        self.wr_reg('TS_PD', 0)

    def power_down_TempSen(self):
        self.wr_reg('TS_PD', 1)

    def check_temp(self, mode = 'bits'):
        # kept for compatibility
        return self.read_temp(mode=mode)

    def read_TempSen_status(self):
        return self.rd_reg("TS_PD")==0

    def read_temp(self, mode = 'bits'):
        if not self.read_TempSen_status():
            if self.verbose:
                print("Sensor was powered down, don't expect valid results")
            self.power_up_TempSen()  # NOTE power up for next time
            return 0
        #C1 and C2 need to be calibrated
        C2 = -0.0073
        C1 = 26
        qoK = 11604.5181
        if self.rb.ver < 3:
            raw = self.rb.SCA.read_adc(self.vtemp, raw=True if mode=='bits' else False)
        else:
            raw = self.rb.MUX64.read_adc(self.vtemp, calibrate=False if mode=='bits' else True)
        #self.power_down_TempSen()  # NOTE this was removed because powering up the temperature sensor can take a considerable amount of time
        if mode.lower() == 'bits' or mode.lower() == 'raw':
            return raw
        elif mode.lower().count('volt'):
            return raw
        elif mode.lower() == 'celsius':
            return (raw - C2)*qoK/C1 - 273.15
        else:
            return (raw - C2)*qoK/C1

    # The TDC clock testing enable.
    # 1'b1: sending TDC clock at the left serial port;
    # 1'b0: sending left serializer data at the left port.
    def enable_TDCClkTest(self):
        self.wr_reg('TDCClockTest', 1)

    def disable_TDCClkTest(self):
        self.wr_reg('TDCClockTest', 0)

    # The TDC reference strobe testing enable.
    # 1'b1: sending TDC reference strobe at the right serial port;
    # 1'b0: sending right serializer data at the right port.
    def enable_TDCRefStrTest(self):
        self.wr_reg('TDCStrobeTest', 1)

    def disable_TDCRefStrTest(self):
        self.wr_reg('TDCStrobeTest', 0)

    # Left/Right Tx amplitude selection.
    # 3'b000: min amplitude (50 mV)
    # 3'b111: max amplitude (320 mV)
    # Step size is about 40 mV.
    def set_TxAmplSel(self, side, amp):
        regs = {'left':'LTx_AmplSel', 'right':'RTx_AmplSel'}
        try:
            self.wr_reg(regs[side], amp)
        except KeyError:
            print('Choose between \'left\' or \'right\' side options')

    def get_TxAmplSel(self, side):
        regs = {'left':'LTx_AmplSel', 'right':'RTx_AmplSel'}
        try:
            return self.rd_reg(regs[side])
        except KeyError:
            print('Choose between \'left\' or \'right\' side options')

    # Left/Right Tx disable, active high.
    def enable_Tx(self, side):
        regs = {'left':'disLTx', 'right':'disRTx'}
        try:
            self.wr_reg(regs[side], 0)
        except KeyError:
            print('Choose between \'left\' or \'right\' side options')

    def disable_Tx(self, side):
        regs = {'left':'disLTx', 'right':'disRTx'}
        try:
            self.wr_reg(regs[side], 1)
        except KeyError:
            print('Choose between \'left\' or \'right\' side options')

    # GRO TOA reset, active low
    def set_GROTOAReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 0 if reset else 1
        self.wr_reg('GRO_TOARST_N', val)

    def get_GROTOAReset(self):
        return self.rd_reg('GRO_TOARST_N')

    # GRO Start, active high.
    def start_GRO(self):
        self.wr_reg('GRO_Start', 1)

    def stop_GRO(self):
        self.wr_reg('GRO_Start', 0)

    # GRO TOA latch clock. (Guessing this means enable/disable)
    def enable_GROTOALatch(self):
        self.wr_reg('GRO_TOA_Latch', 1)

    def disable_GROTOALatch(self):
        self.wr_reg('GRO_TOA_Latch', 0)

    # GRO TOA clock.
    def enable_GROTOAClk(self):
        self.wr_reg('GRO_TOA_CK', 1)

    def disable_GROTOAClk(self):
        self.wr_reg('GRO_TOA_CK', 0)

    # GRO TOT clock.
    def enable_GROTOTClk(self):
        self.wr_reg('GRO_TOT_CK', 1)

    def disable_GROTOTClk(self):
        self.wr_reg('GRO_TOT_CK', 0)

    # GRO TOT reset, active low.
    def set_GROTOTReset(self, reset):
        if not isinstance(reset, bool):
            raise TypeError('Argument must be True (reset) or False (don\'t reset)')
        val = 0 if reset else 1
        self.wr_reg('GRO_TOTRST_N', val)

    def get_GROTOTReset(self):
        return self.rd_reg('GRO_TOTRST_N')

    # ***********************
    # **** PERIPH STATUS ****
    # ***********************

    # Bit alignment error
    def get_BitAlignErr(self):
        return self.rd_reg('fcBitAlignError')

    # Phase shifter late
    def get_PhaseShiftLate(self):
        return self.rd_reg('PS_Late')

    # AFC capacitance
    def get_AFCCap(self):
        return self.rd_reg('AFCcalCap')

    # AFC busy, 1: AFC is ongoing, 0: AFC is done
    def get_AFCBusy(self):
        return self.rd_reg('AFCBusy')

    # Fast command alignment FSM state
    def get_FSM_FCAlign(self):
        return self.rd_reg('fcAlignFinalState')

    # Global control FSM state
    def get_FSM_GlobCtrl(self):
        return self.rd_reg('controllerState')

    # Fast command self-alignment error indicator, ed[3:0] in figure 53
    def get_SelfAlignErr(self):
        return self.rd_reg('fcAlignStatus')

    # Count of invalid fast command received
    def get_invalidFCCount(self):
        return self.rd_reg('invalidFCCount')

    def FC_status(self):
        status = self.get_invalidFCCount() == self.invalid_FC_counter
        self.invalid_FC_counter = self.get_invalidFCCount()
        return status

    # Count of PLL unlock detected
    def get_PLLUnlockCount(self):
        return self.rd_reg('pllUnlockCount')

    # 32-bit EFuse output
    def get_EFuseOut(self):
        return self.rd_reg('EFuseQ')
