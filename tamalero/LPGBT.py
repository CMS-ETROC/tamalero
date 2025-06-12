from tamalero.RegParser import RegParser
import os
import sys
import pickle
import copy
import random
import json
from functools import wraps
import tamalero.colors as colors
from tamalero.colors import red, green
from tamalero.utils import read_mapping, chunk, load_yaml, get_config, majority_vote
from time import sleep
from datetime import datetime
try:
    from tabulate import tabulate
    has_tabulate = True
except ModuleNotFoundError:
    print ("Package `tabulate` not found.")
    has_tabulate = False

from tamalero.lpgbt_constants import LpgbtConstants

def gpio_byname(gpio_func):
    @wraps(gpio_func)
    def wrapper(lpgbt, pin, direction=1):
        if isinstance(pin, str):
            gpio_dict = lpgbt.gpio_mapping
            pin = gpio_dict[pin]['pin']
            return gpio_func(lpgbt, pin, direction)
        elif isinstance(pin, int):
            return gpio_func(lpgbt, pin, direction)
        else:
            invalid_type = type(pin)
            raise TypeError(f"{gpio_func.__name__} can only take positional arguments of type int or str, but argument of type {invalid_type} was given.")

    return wrapper

class LPGBT(RegParser):

    def __init__(self, rb=0, trigger=False, flavor='small', master=None, kcu=None, do_adc_calibration=False, config='default', debug=False, ver=None, verbose=False, poke=False, rbver=None):
        '''
        Initialize lpGBT for a certain readout board number (rb).
        The trigger lpGBT is accessed through I2C of the master (= DAQ lpGBT).
        '''
        self.nodes = {}
        self.rb = rb
        self.trigger = trigger
        self.calibrated = False
        self.gain = 1.85
        self.offset = 512
        self.verbose = verbose
        if ver is not None:
            self.ver = ver
        if rbver is None:
            self.rbver = self.ver + 1
        else:
            self.rbver = rbver

        if self.trigger:
            assert isinstance(master, LPGBT), "Trying to initialize a trigger lpGBT but got no lpGBT master."
            self.master = master
        self.LPGBT_CONST = LpgbtConstants()

        if kcu != None:
            self.kcu = kcu

        self.config = config
        if not debug and not poke:
            self.configure(do_adc_calibration=do_adc_calibration)
        else:
            if debug:
                print("Warning: Initializing lpGBT in debug mode.")
            if poke:
                self.ver = 1  # hard coded for now
            self.kcu.write_node("READOUT_BOARD_%d.SC.FRAME_FORMAT" % self.rb, self.ver)
            self.parse_xml(ver=self.ver)

        if self.rbver is None:
            print("RB version could not be correctly identified by LPGBT")


    def configure(self, do_adc_calibration=True):
        if not hasattr(self, 'kcu'):
            raise Exception("Connect to KCU first.")

        if self.trigger:
            self.ver = self.master.ver
            self.serial_num = self.master.serial_num
            #return

        if self.kcu.dummy:
            self.ver = 0
            return

        # Get LPGBT Version
        timeout = 0
        if not hasattr(self, 'ver'):
            if self.verbose:
                print ("Figuring out lpGBT version by reading from ROMREG")
            while True:
                # https://lpgbt.web.cern.ch/lpgbt/v0/registermap.html#x1c5-rom
                # Writing to addresses directly because readback will still fail here
                self.kcu.write_node("READOUT_BOARD_%d.SC.FRAME_FORMAT" % self.rb, 0)
                # toggle the uplink to and from 40MHz clock, for some reason this is
                # needed for the mgt to lock
                self.wr_adr(0x118, 0xC0) # https://lpgbt.web.cern.ch/lpgbt/v0/registermap.html#x118-uldatasource0
                sleep(0.01)
                self.wr_adr(0x118, 0) # https://lpgbt.web.cern.ch/lpgbt/v0/registermap.html#x118-uldatasource0
                if self.rb == 0 and self.trigger:
                    self.wr_adr(0x036, 0x00)
                else:
                    self.wr_adr(0x036, 0x80)  # we might want to go back to the inversion with the next FW version
                self.wr_adr(0x0ef, 0x6)
                sleep(0.01)
                is_v0 = (self.rd_adr(0x1c5) == 0xa5)

                # https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#x1d7-rom
                self.kcu.write_node("READOUT_BOARD_%d.SC.FRAME_FORMAT" % self.rb, 1)
                self.wr_adr(0x128, 0xC0) # https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#x128-uldatasource0
                sleep(0.01)
                self.wr_adr(0x128, 0) # https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#x128-uldatasource0
                if self.rb == 0 and self.trigger:
                    self.wr_adr(0x036, 0x00)
                else:
                    self.wr_adr(0x036, 0x80)  # we might want to go back to the inversion with the next FW version
                #self.wr_adr(0x036, 0x80)
                self.wr_adr(0x0fb, 0x6)
                sleep(0.01)
                is_v1 = (self.rd_adr(0x1d7) == 0xa6)

                is_v2 = (self.rd_adr(0x1d7) == 0xae)
                
                print("0x1d7 readback value is", f'0x{(self.rd_adr(0x1d7)):02x}')

                print("lpGBT version found", is_v0 ^ is_v1 ^ is_v2)

                if is_v0 ^ is_v1 ^ is_v2:
                    break
                self.reset_daq_mgts()
                sleep(0.05)
                timeout += 1
                print(timeout)
                if timeout > 50:
                    raise Exception("Could not successfully read from lpGBT and failed to determine lpGBT version. Check optical links and power of RB.")

            if is_v0 and not is_v1 and not is_v2:
                print (" > lpGBT v0 detected")
                self.ver = 0
            elif is_v1 and not is_v0 and not is_v2:
                print (" > lpGBT v1 detected")
                self.ver = 1
            elif is_v2 and not is_v0 and not is_v1:
                print (" > lpGBT v2 detected")
                self.ver = 1
            else:
                print (" > unsure about lpGBT version. This case should have been impossible to reach.")
                raise Exception("Spurious lpGBT version.")

            #     if is_v0 ^ is_v1:
            #         break
            #     self.reset_daq_mgts()
            #     sleep(0.05)
            #     timeout += 1
            #     if timeout > 50:
            #         raise Exception("Could not successfully read from lpGBT and failed to determine lpGBT version. Check optical links and power of RB.")

            # if is_v0 and not is_v1:
            #     print (" > lpGBT v0 detected")
            #     self.ver = 0
            # elif is_v1 and not is_v0:
            #     print (" > lpGBT v1 detected")
            #     self.ver = 1
            # else:
            #     print (" > unsure about lpGBT version. This case should have been impossible to reach.")
            #     raise Exception("Spurious lpGBT version.")

        if self.rbver is None:
            self.rbver = self.ver + 1

        #self.base_config = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/lpgbt_config.yaml'))['base'][f'v{self.ver}']
        #self.ec_config = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/lpgbt_config.yaml'))['ec'][f'v{self.ver}']

        self.base_config = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/lpgbt_smu_config.yaml'))['base'][f'v{self.ver}']
        self.ec_config = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/lpgbt_smu_config.yaml'))['ec'][f'v{self.ver}']

        self.kcu.write_node("READOUT_BOARD_%d.SC.FRAME_FORMAT" % self.rb, self.ver)
        self.parse_xml(ver=self.ver)

        if self.trigger:
            self.init_trigger_links()
            if self.rb == 0 and False:
                self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x0)  # this is already done for v1
            else:
                self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x1)  # this is already done for v1
            sleep(0.01)
            self.wr_reg("LPGBT.RWF.POWERUP.DLLCONFIGDONE", 0x1)  # NOTE untested change
            self.wr_reg("LPGBT.RWF.POWERUP.PLLCONFIGDONE", 0x1)

        self.set_adc_mapping()
        self.set_gpio_mapping()

        # Get LPGBT Serial Num
        self.serial_num = 0# self.get_board_id()['lpgbt_serial']

        self.link_inversions = get_config(self.config, version=f'v{self.ver+1}')['inversions']

        if not self.power_up_done():
            print(" > Running power up within LPGBT.configure()")
            self.power_up_init()

        self.invert_links()

        self.set_dac(1.0)  # set the DAC / Vref to 1.0V.
        # Callibrate ADCs
        # will automatically load from the config file if it is found
        if do_adc_calibration and not self.calibrated:
            self.calibrate_adc()

        #self.current_adcs = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/current_adcs.yaml'))['lpGBT']
        #for adc in self.current_adcs:
        #    self.set_current_adc(adc)

    def read_base_config(self):
        #
        print("{:80}{:10}{:10}".format("Register", "value", "default"))
        for reg in self.base_config:
            res = self.rd_reg(reg)
            colored = green if res == self.base_config[reg] else red
            print (colored("{:80}{:<10}{:<10}".format(reg, res, self.base_config[reg])))

    def base_configuration(self):
        # this could be extended to run over the configuration in lpgbt_config.yaml file
        # BUT this still needs to be tested. the two settings below seems to be enough for now
        # NOTE: maybe some config is still missing for proper SCA communication for lpGBT v1
        if self.trigger and self.rb==0 and False:
            self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x0)  # this is already done for v1
        else:
            self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x1)  # this is already done for v1
        self.wr_reg("LPGBT.RWF.POWERUP.DLLCONFIGDONE", 0x1)  # NOTE untested change
        self.wr_reg("LPGBT.RWF.POWERUP.PLLCONFIGDONE", 0x1)

    def set_adc_mapping(self):
        assert self.rbver in [1,2,3], f"Unrecognized version {self.rbver}"
        self.adc_mapping = get_config(self.config, version=f'v{self.rbver}')['LPGBT']['adc']
        for channel in self.adc_mapping:
            if self.adc_mapping[channel]['pin'] < 8:  # ignore internal channels
                if self.adc_mapping[channel]['current'] == 1:
                    if self.verbose:
                        print(f'Enabling current soure for ADC{channel}')
                    self.set_current_adc(self.adc_mapping[channel]['pin'])
                else:
                    if self.verbose:
                        print(f'Disabling current soure for ADC{channel}')
                    self.set_current_adc(self.adc_mapping[channel]['pin'], to=0)
        #if self.ver == 0:
        #    self.adc_mapping = read_mapping(os.path.expandvars('$TAMALERO_BASE/configs/LPGBT_mapping.yaml'), 'adc')
        #elif self.ver == 1:
        #    self.adc_mapping = read_mapping(os.path.expandvars('$TAMALERO_BASE/configs/LPGBT_mapping_v2.yaml'), 'adc')

    def set_gpio_mapping(self):
        assert self.rbver in [1,2,3], f"Unrecognized version {self.rbver}"
        if self.rbver > 2 and self.trigger:
            self.gpio_mapping = get_config(self.config, version=f'v{self.rbver}')['LPGBT2']['gpio']
        else:
            self.gpio_mapping = get_config(self.config, version=f'v{self.rbver}')['LPGBT']['gpio']
        #if self.ver == 0:
        #    self.gpio_mapping = read_mapping(os.path.expandvars('$TAMALERO_BASE/configs/LPGBT_mapping.yaml'), 'gpio')
        #elif self.ver == 1:
        #    self.gpio_mapping = read_mapping(os.path.expandvars('$TAMALERO_BASE/configs/LPGBT_mapping_v2.yaml'), 'gpio')

    def update_rb_ver(self, new_ver):
        assert new_ver in [1,2,3], f"Unrecognized version {new_ver}"
        self.rbver = new_ver
        self.set_adc_mapping()
        self.set_gpio_mapping()
        self.configure_gpios()

    def link_status(self):
        if self.trigger:
            if self.verbose:
                print ("Checking trigger link status")
                print ("Uplink ready:", self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_1.READY"%self.rb).value()==1 )
                print ("FEC count:", self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_1.FEC_ERR_CNT"%self.rb).value())
            return (
                (self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_1.FEC_ERR_CNT"%self.rb).value() == 0) &
                (self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_1.READY"%self.rb).value() == 1)
            )
        else:
            if self.verbose:
                print ("Checking DAQ link status")
                print ("Uplink ready:", self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_0.READY"%self.rb).value()==1 )
                print ("FEC count:", self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_0.FEC_ERR_CNT"%self.rb).value())
            return (
                (self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_0.FEC_ERR_CNT"%self.rb).value() == 0) &
                (self.kcu.read_node("READOUT_BOARD_%i.LPGBT.UPLINK_0.READY"%self.rb).value() == 1)
            )

    def get_version(self):
        self.ver = self.get_board_id()['lpgbt_ver']
        return self.ver
        #self.ver = self.rd_reg("LPGBT.RWF.CHIPID.USERID1") & 1
        #self.ver = self.rd_adr(0x005).value() & 1

    def reset_tx_mgt_by_mask(self, mask):
        id = "SYSTEM.MGT_TX_RESET"
        self.kcu.write_node(id, mask)

    def reset_rx_mgt_by_mask(self, mask):
        id = "SYSTEM.MGT_RX_RESET"
        self.kcu.write_node(id, mask)

    def reset_trigger_mgts(self):
        id = "SYSTEM.MGT_RX_RESET"
        # trigger links on 1,3,5,7,9
        self.kcu.write_node(id, 0x2aa)
        self.kcu.write_node(id, 0x000)

    def reset_daq_mgts(self):

        for id in ["SYSTEM.MGT_RX_RESET", "SYSTEM.MGT_TX_RESET"]:
            # daq links on 0,2,4,6,8
            self.kcu.write_node(id, 0x155)
            self.kcu.write_node(id, 0x000)

    def reset(self):
        self.wr_reg("LPGBT.RW.RESET.RSTPLLDIGITAL", 1)
        self.wr_reg("LPGBT.RW.RESET.RSTFUSES",      1)
        self.wr_reg("LPGBT.RW.RESET.RSTRXLOGIC",    1)
        self.wr_reg("LPGBT.RW.RESET.RSTTXLOGIC",    1)

        self.wr_reg("LPGBT.RW.RESET.RSTPLLDIGITAL", 0)
        self.wr_reg("LPGBT.RW.RESET.RSTFUSES",      0)
        self.wr_reg("LPGBT.RW.RESET.RSTRXLOGIC",    0)
        self.wr_reg("LPGBT.RW.RESET.RSTTXLOGIC",    0)

    def init_trigger_links(self):
        if self.trigger:
            lpgbt = self.master
        else:
            lpgbt = self

        id = "LPGBT.RW.TESTING.ULECDATASOURCE"
        node = lpgbt.get_node(id)
        self.write_reg(self.master.I2C_write, self.master.I2C_read, node, value=6)
        sleep (0.1)
        self.write_reg(self.master.I2C_write, self.master.I2C_read, node, value=0)
        #lpgbt.I2C_write(reg=0x118, val=6, master=2, slave_addr=0x70)
        #sleep (0.1)
        #lpgbt.I2C_write(reg=0x118, val=0, master=2, slave_addr=0x70)

        self.reset_trigger_mgts()

    def power_up_init(self):
        if not self.trigger:

            # toggle the uplink to and from 40MHz clock, for some reason this is
            # needed for the mgt to lock

            if (not self.kcu.read_node(
                    "READOUT_BOARD_%d.LPGBT.UPLINK_0.READY" % self.rb)):
                print("  > Performing LpGBT Magic...")
                id = "LPGBT.RW.TESTING.ULECDATASOURCE"
                self.wr_reg(id, 6)
                sleep(0.1)
                self.wr_reg(id, 0)
                print("  > Magic Done")
        else:
            # servant lpgbt base configuration
            self.init_trigger_links()
            sleep(0.1)

            if self.ver == 0:
                self.master.program_slave_from_file(os.path.expandvars('$TAMALERO_BASE/configs/config_slave.txt'))
            elif self.ver == 1:
                self.master.program_slave_from_file(os.path.expandvars('$TAMALERO_BASE/configs/config_slave_v1.txt'))
            sleep(0.1)
            self.invert_links()

            # toggle the uplink to and from 40MHz clock, for some reason this is
            # needed for the mgt to lock

        self.base_configuration()

        if not self.trigger:
            self.configure_gpios()
            self.initialize()
            self.config_eport_dlls()
            self.configure_eptx()
            self.configure_eprx()
        elif self.rbver > 2 and self.trigger:
            self.configure_gpios()

        self.set_power_up_done()


    def connect_KCU(self, kcu):
        '''
        We need to connect to the KCU somehow
        '''
        self.kcu = kcu

    def align_DAQ(self):
        for i in range(28):
            id = "READOUT_BOARD_%d.LPGBT.UPLINK_0.ALIGN_%d" % (self.rb, i)
            self.kcu.write_node(id, 2)

    def wr_adr(self, adr, data):

        if self.trigger:
            return self.master.I2C_write(adr, data)
            #raise NotImplementedError("rd_adr does only read from the master lpGBT, and you're trying to write to a servant")
        else:
            self.kcu.toggle_dispatch()
            #self.kcu.write_node("READOUT_BOARD_%d.SC.TX_GBTX_ADDR" % self.rb, 115)
            self.kcu.write_node("READOUT_BOARD_%d.SC.TX_REGISTER_ADDR" % self.rb, adr)
            self.kcu.write_node("READOUT_BOARD_%d.SC.TX_DATA_TO_GBTX" % self.rb, data)
            self.kcu.action("READOUT_BOARD_%d.SC.TX_WR" % self.rb)
            self.kcu.action("READOUT_BOARD_%d.SC.TX_START_WRITE" % self.rb)
            self.kcu.dispatch()

    def rd_adr(self, adr):
        if self.trigger:
            return self.master.I2C_read(adr)
            #raise NotImplementedError("rd_adr does only read from the master lpGBT, and you're trying to read from a servant")
        else:
            self.kcu.toggle_dispatch()
            self.kcu.write_node("READOUT_BOARD_%d.SC.TX_REGISTER_ADDR" % self.rb, adr)
            self.kcu.dispatch()
            self.kcu.action("READOUT_BOARD_%d.SC.TX_START_READ" % self.rb)
            valid = self.kcu.read_node("READOUT_BOARD_%d.SC.RX_DATA_VALID" % self.rb).valid()
            if valid:
                # this only means that the KCU successfully read data
                # not necessarily does it mean there's communication with the lpGBT
                return self.kcu.read_node("READOUT_BOARD_%d.SC.RX_DATA_FROM_GBTX" % self.rb)

            print("LpGBT read failed!")
            return None

    def wr_reg(self, id, data):
        node = self.get_node(id)
        if self.trigger:
            self.write_reg(self.master.I2C_write, self.master.I2C_read, node, data)  # inherited from RegParser
        else:
            self.write_reg(self.wr_adr, self.rd_adr, node, data)  # inherited from RegParser

    def rd_reg(self, id):
        node = self.get_node(id)
        if self.trigger:
            data = self.read_reg(self.master.I2C_read, node)  # inherited from RegParser
        else:
            data = self.read_reg(self.rd_adr, node)
        return data

    def rd_flush(self):
        i = 0
        while (not self.kcu.read_node("READOUT_BOARD_%d.SC.RX_EMPTY" % self.rb)):
            self.kcu.action("READOUT_BOARD_%d.SC.RX_RD" % self.rb)
            read = self.kcu.read_node("READOUT_BOARD_%d.SC.RX_DATA_FROM_GBTX" % self.rb)
            i = i + 1

#    def configure_gpio_outputs(self, outputs=0x2401, defaults=0x0401):
#        # NOTE: v0: defaults = 0x0401, outputs = 0x2401  (Rhett LED off)
#        #       v1: defaults = 0x0409, outputs = 0x2409  (Rhett LED on)
#        # have to first set defaults, then switch to outputs otherwise we reset the VTRx+
#        self.wr_reg('LPGBT.RWF.PIO.PIOOUTH', defaults >> 8)
#        self.wr_reg('LPGBT.RWF.PIO.PIOOUTL', defaults & 0xFF)
#        self.wr_reg('LPGBT.RWF.PIO.PIODIRH', outputs >> 8)
#        self.wr_reg('LPGBT.RWF.PIO.PIODIRL', outputs & 0xFF)

#    def gpio_byname(self, gpio_func):
#        @wraps(gpio_func)
#        def wrapper(*args, **kwargs):
#            if all([type(arg) == str for arg in args]):
#                gpio_dict = self.gpio_mapping
#                pin = gpio_dict[list(args)[0]]['pin']
#                return gpio_func(pin, **kwargs)
#            elif all([type(arg) == int for arg in args]):
#                return gpio_func(*args, **kwargs)
#            else:
#                invalid_type = type(list(args)[0])
#                raise TypeError(f"{gpio_func.__name__} can only take positional arguments of type int or str, but argument of type {invalid_type} was given.")
#
#        return wrapper

    def configure_gpios(self): #read and print all adc values
        gpio_dict = self.gpio_mapping
        if self.verbose:
            print("Configuring LPGBT GPIO Pins...")
        for gpio_reg in gpio_dict.keys():
            pin         = gpio_dict[gpio_reg]['pin']
            direction   = int(gpio_dict[gpio_reg]['direction'] == 'out')
            comment     = gpio_dict[gpio_reg]['comment']
            default     = gpio_dict[gpio_reg]['default']
            if self.verbose:
                print("Setting LPGBT GPIO pin %s (%s) to %s"%(pin, comment, gpio_dict[gpio_reg]['direction']))
            self.set_gpio(pin, default)               # Defaults must be set first
            self.set_gpio_direction(pin, direction)   # Then switch to directions otherwise we reset the VTRx+

    def read_gpio(self, reg, pin):
        val = self.rd_reg(reg)
        return int((val >> pin) & 1)

    @gpio_byname
    def set_gpio(self, pin, value=1):
        assert pin < 16 and pin >= 0

        if pin < 8:
            out_reg = 'LPGBT.RWF.PIO.PIOOUTL'
            read_reg = 'LPGBT.RO.ECLK.PIOINL'
        else:
            out_reg = 'LPGBT.RWF.PIO.PIOOUTH'
            read_reg = 'LPGBT.RO.ECLK.PIOINH'
            pin -= 8

        if self.ver == 1:
            currently_set = self.rd_reg(read_reg)
        else:
            currently_set = self.rd_reg(out_reg)

        if (currently_set & (1 << pin)) and value==0:
            currently_set ^= (1 << pin)
        elif value==1:
            currently_set |= (1 << pin)

        self.wr_reg(out_reg, currently_set)
        return self.read_gpio(out_reg, pin)  # in order to check it is actually set

    @gpio_byname
    def set_gpio_direction(self, pin, direction=1):
        assert pin < 16 and pin >= 0

        if pin < 8:
            dir_reg = 'LPGBT.RWF.PIO.PIODIRL'
        else:
            dir_reg = 'LPGBT.RWF.PIO.PIODIRH'
            pin -= 8

        currently_set = self.rd_reg(dir_reg)

        if (currently_set & (1 << pin)) and direction==0:
            currently_set ^= (1 << pin)
        elif direction==1:
            currently_set |= (1 << pin)

        self.wr_reg(dir_reg, currently_set)
        return self.read_gpio(dir_reg, pin)  # in order to check it is actually set

    def set_uplink_alignment(self, link, val, quiet=False):
        if self.trigger:
            if not quiet:
                print ("Setting uplink alignment for trigger link %i to %i"%(link, val))
            id = "READOUT_BOARD_%d.LPGBT.UPLINK_1.ALIGN_%d" % (self.rb, link)
        else:
            if not quiet:
                print ("Setting uplink alignment for DAQ link %i to %i"%(link, val))
            id = "READOUT_BOARD_%d.LPGBT.UPLINK_0.ALIGN_%d" % (self.rb, link)
        self.kcu.write_node(id, val)

    def get_uplink_alignment(self, link):
        if self.trigger:
            return self.kcu.read_node("READOUT_BOARD_%d.LPGBT.UPLINK_1.ALIGN_%d"%(self.rb, link)).value()
        else:
            return self.kcu.read_node("READOUT_BOARD_%d.LPGBT.UPLINK_0.ALIGN_%d"%(self.rb, link)).value()

    def set_uplink_invert(self, link, invert=True):
        self.wr_reg("LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX%dINVERT" % link, invert)

    #def set_downlink_invert(self, link, invert=True):
    #    self.wr_reg("LPGBT.RWF.EPORTTX.EPTX{:02d}INVERT".format(link), invert)

    def set_downlink_invert(self, link, invert=True):
        group = link // 4
        elink = link % 4
        self.wr_reg("LPGBT.RWF.EPORTTX.EPTX%d%dINVERT" % (group, elink), invert)

    def set_clock_invert(self, link, invert=True):
        self.wr_reg("LPGBT.RWF.EPORTCLK.EPCLK%dINVERT" % link, invert)

    def invert_links(self):
        if self.trigger:
            for link in range(28):
                self.set_uplink_invert(link, invert=False)
        else:
            for link in range(28):
                self.set_uplink_invert(link, invert=False)
            #for link in [0,2,10,20,22,30]:
            for link in [0,2,4,8,10,12]:
                self.set_downlink_invert(link, invert=False)
            for link in [0,1,2,3,4,5,22,23,24,25,26,27]:
                self.set_clock_invert(link, invert=False)
        if self.trigger:
            for link in self.link_inversions['trigger']:
                self.set_uplink_invert(link)
        else:
            for link in self.link_inversions['clocks']:
                self.set_clock_invert(link)
            for link in self.link_inversions['downlink']:
                self.set_downlink_invert(link)
            for link in self.link_inversions['uplink']:
                self.set_uplink_invert(link)

    def read_inversions(self):
        if self.trigger:
            print("Trigger LPGBT Registers -- Uplinks")
            for link in self.link_inversions['trigger']:
                register = "LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX%dINVERT" % link
                val = self.rd_reg(register)
                print(register, "\t", val)
        else:
            print("DAQ LPGBT Registers -- Clocks")
            for link in self.link_inversions['clocks']:
                register = "LPGBT.RWF.EPORTCLK.EPCLK%dINVERT" % link
                val = self.rd_reg(register)
                print(register, "\t", val)
            print("DAQ LPGBT Registers -- Downlinks")
            for link in self.link_inversions['downlink']:
                group = link // 4
                elink = link % 4
                register = "LPGBT.RWF.EPORTTX.EPTX%d%dINVERT" % (group, elink)
                val = self.rd_reg(register)
                print(register, "\t", val)
            print("DAQ LPGBT Registers -- Uplinks")
            for link in self.link_inversions['uplink']:
                register = "LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX%dINVERT" % link
                val = self.rd_reg(register)
                print(register, "\t", val)

    def get_uplink_invert(self, link):
        return self.rd_reg("LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX%dINVERT" % link)

        #if self.trigger:
        #    return self.I2C_read(reg=0xcc+link, master=2, slave_addr=0x70)
        #else:
        #    return self.rd_adr(0xcc+link).value()

    def configure_clocks(self, en_mask):
        for i in range(28):
            if 0x1 & (en_mask >> i):
                self.wr_reg("LPGBT.RWF.EPORTCLK.EPCLK%dFREQ" % i, 1)
                self.wr_reg("LPGBT.RWF.EPORTCLK.EPCLK%dDRIVESTRENGTH" % i, 4)

    def config_eport_dlls(self):
        if self.verbose:
            print("Configuring eport dlls...")
        if self.ver == 0:
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXDLLCURRENT", 0x1)
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXDLLCONFIRMCOUNT", 0x1)
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXDLLFSMCLKALWAYSON", 0x0)
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXDLLCOARSELOCKDETECTION", 0x0)
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXENABLEREINIT", 0x0)
            self.wr_reg("LPGBT.RWF.CLOCKGENERATOR.EPRXDATAGATINGENABLE", 0x1)
        elif self.ver == 1:
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXDLLCURRENT", 0x1)
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXDLLCONFIRMCOUNT", 0x1)
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXDLLFSMCLKALWAYSON", 0x0)
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXDLLCOARSELOCKDETECTION", 0x0)
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXENABLEREINIT", 0x0)
            self.wr_reg("LPGBT.RWF.EPORTRX.EPRXDATAGATINGDISABLE", 0x0)

    def configure_eptx(self):
        for i in range(4):
            self.wr_reg(f"LPGBT.RWF.EPORTTX.EPTX{i}DATARATE", 0x3)

        tx_chn_indices = [0, 2, 4, 8, 10, 12]
        groups = list(set([i // 4 for i in tx_chn_indices]))
        groups.sort()

        for group in groups:
            for link in range(3, -1, -1):
                reg_name = f"LPGBT.RWF.EPORTTX.EPTX{group}{link}ENABLE"
                self.wr_reg(reg_name, 0x1)

                i = group * 4 + link
                self.wr_reg(f"LPGBT.RWF.EPORTTX.EPTX_CHN_CONTROL.EPTX{i}DRIVESTRENGTH", 0x3)
                self.wr_reg(f"LPGBT.RWF.EPORTTX.EPTX_CHN_CONTROL.EPTX{i}PREEMPHASISSTRENGTH", 0x0)
                self.wr_reg(f"LPGBT.RWF.EPORTTX.EPTX_CHN_CONTROL.EPTX{i}PREEMPHASISMODE", 0x0)

                if self.verbose:
                    print(f"LPGBT.RWF.EPORTTX.EPTX{group}{link}ENABLE")

        for i in range(4):
            self.wr_reg(f"LPGBT.RWF.EPORTTX.EPTX{i}MIRRORENABLE", 0x1)

    def configure_eprx(self):
        if self.verbose:
            print("Configuring elink inputs...")

        #rx_chn_indices = [0, 2, 4, 6, 8, 10, 12, 14]
        rx_chn_indices = [0, 4, 8, 12] # etroc2 right output selected
        groups = list(set([i // 4 for i in rx_chn_indices]))
        groups.sort()

        for group in groups:
            self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX{group}DATARATE", 1)  # 1 for 320 mbps and 2 for 640 Mbps
            self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX{group}TRACKMODE", 2)  # continuous phase tracking

            for link in range(3, -1, -1):
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX{group}{link}ENABLE", 0x1)

                i = group * 4 + link
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{i}PHASESELECT", 5)
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{i}INVERT", 0)
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{i}ACBIAS", 1)
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{i}TERM", 1)
                self.wr_reg(f"LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX{i}EQ1", 0)

        self.wr_reg("LPGBT.RWF.EPORTRX.EPRXLOCKTHRESHOLD", 0x5)
        self.wr_reg("LPGBT.RWF.EPORTRX.EPRXRELOCKTHRESHOLD", 0x5)

    # def configure_eptx(self):

    #     for i in range(4):
    #         # [0x0a7] EPTXDataRate
    #         self.wr_reg("LPGBT.RWF.EPORTTX.EPTX%dDATARATE" % i, 0x3)

    #     #self.wr_reg("LPGBT.RWF.EPORTTX.EPTX00INVERT" , 0x1)

    #     # EPTXxxEnable
    #     # EPTXxxDriveStrength
    #     for i in [0, 2, 4, 8, 10, 12]:
    #         group = str(i//4)
    #         link = str(i % 4)
    #         self.wr_reg("LPGBT.RWF.EPORTTX.EPTX%s%sENABLE" % (group, link), 0x1)
    #         self.wr_reg("LPGBT.RWF.EPORTTX.EPTX_CHN_CONTROL.EPTX%dDRIVESTRENGTH" % i, 0x3)
    #         if self.verbose:
    #             print("LPGBT.RWF.EPORTTX.EPTX%s%sENABLE" % (group, link))

    #     # enable mirror feature
    #     for i in range(4):
    #         self.wr_reg("LPGBT.RWF.EPORTTX.EPTX%dMIRRORENABLE" % i, 0x1)

    # def configure_eprx(self):

    #     if self.verbose:
    #         print("Configuring elink inputs...")
    #     # Enable Elink-inputs

    #     # enable inputs
    #     for i in range(24):
    #         group = str(i//4)
    #         link = str(i % 4)
    #         self.wr_reg("LPGBT.RWF.EPORTRX.EPRX%s%sENABLE" % (group, link), 1)

    #     for i in range(7):
    #         # set banks to 320 Mbps (1) or 640 Mbps (2)
    #         self.wr_reg("LPGBT.RWF.EPORTRX.EPRX%dDATARATE" % i, 1)
    #         # set banks to continuous phase tracking (2)
    #         self.wr_reg("LPGBT.RWF.EPORTRX.EPRX%dTRACKMODE" % i, 2)

    #     # enable 100 ohm termination
    #     for i in range(28):
    #         self.wr_reg("LPGBT.RWF.EPORTRX.EPRX_CHN_CONTROL.EPRX%dTERM" % i, 1)

    def init_adc(self):
        self.wr_reg("LPGBT.RW.ADC.ADCENABLE", 0x1)  # enable ADC
        self.wr_reg("LPGBT.RW.ADC.TEMPSENSRESET", 0x1)  # resets temp sensor
        self.wr_reg("LPGBT.RW.ADC.VDDMONENA", 0x1)  # enable dividers
        self.wr_reg("LPGBT.RW.ADC.VDDTXMONENA", 0x1)  # enable dividers
        self.wr_reg("LPGBT.RW.ADC.VDDRXMONENA", 0x1)  # enable dividers
        if self.ver == 0:
            self.wr_reg("LPGBT.RW.ADC.VDDPSTMONENA", 0x1,)  # enable dividers
        else:
            self.wr_reg("LPGBT.RW.ADC.VDDMONENA", 0x1,)  # enable dividers
        self.wr_reg("LPGBT.RW.ADC.VDDANMONENA", 0x1)  # enable dividers
        self.wr_reg("LPGBT.RWF.CALIBRATION.VREFENABLE", 0x1)  # vref enable
        self.wr_reg("LPGBT.RWF.CALIBRATION.VREFTUNE", 0x63)

    #def read_adcs(self):
    #    self.init_adc()
    #    print("ADC Readings:")
    #    for i in range(16):
    #        name = ""
    #        conv = 0
    #        if (i==0 ): conv=1;      name="VTRX TH1"
    #        if (i==1 ): conv=1/0.55; name="1V4D * 0.55"
    #        if (i==2 ): conv=1/0.55; name="1V5A * 0.55"
    #        if (i==3 ): conv=1/0.33; name="2V5TX * 0.33"
    #        if (i==4 ): conv=1;      name="RSSI"
    #        if (i==5 ): conv=1;      name="N/A"
    #        if (i==6 ): conv=1/0.33; name="2V5RX * 0.33"
    #        if (i==7 ): conv=1;      name="RT1"
    #        if (i==8 ): conv=1;      name="EOM DAC (internal signal)"
    #        if (i==9 ): conv=1/0.42; name="VDDIO * 0.42 (internal signal)"
    #        if (i==10): conv=1/0.42; name="VDDTX * 0.42 (internal signal)"
    #        if (i==11): conv=1/0.42; name="VDDRX * 0.42 (internal signal)"
    #        if (i==12): conv=1/0.42; name="VDD * 0.42 (internal signal)"
    #        if (i==13): conv=1/0.42; name="VDDA * 0.42 (internal signal)"
    #        if (i==14): conv=1;      name="Temperature sensor (internal signal)"
    #        if (i==15): conv=1/0.50; name="VREF/2 (internal signal)"
    #
    #        read = self.read_adc(i)
    #        print("\tch %X: 0x%03X = %f, reading = %f (%s)" % (i, read, read/1024., conv*read/1024., name))

    def read_adcs(self, check=False, strict_limits=False): #read and print all adc values
        self.init_adc()
        adc_dict = self.adc_mapping
        table = []
        will_fail = False
        for adc_reg in adc_dict.keys():
            pin = adc_dict[adc_reg]['pin']
            comment = adc_dict[adc_reg]['comment']
            value = self.read_adc(pin)
            value_raw = self.read_adc(pin, calibrate=False)
            #value_calibrated = value * self.cal_gain / 1.85 + (512 - self.cal_offset)  # FIXME this was applying twice
            input_voltage_direct = value / (2**10 - 1)
            input_voltage = input_voltage_direct * adc_dict[adc_reg]['conv']
            if check:
                try:
                    min_v = adc_dict[adc_reg]['min']
                    max_v = adc_dict[adc_reg]['max']
                    status = "OK" if (input_voltage >= min_v) and (input_voltage <= max_v) else "ERR"
                    if status == "ERR" and strict_limits:
                        will_fail = True
                except KeyError:
                    status = "N/A"
                table.append([adc_reg, pin, value_raw, value, input_voltage_direct, input_voltage, status, comment])
            else:
                table.append([adc_reg, pin, value_raw, value, input_voltage_direct, input_voltage, comment])

        if check:
            headers = ["Register","Pin", "Reading (raw)", "Reading (calib)", "Voltage (direct)", "Voltage (conv)", "Status", "Comment"]
        else:
            headers = ["Register","Pin", "Reading (raw)", "Reading (calib)", "Voltage (direct)", "Voltage (conv)", "Comment"]

        if has_tabulate:
            print(tabulate(table, headers=headers,  tablefmt="simple_outline"))
        else:
            header_string = "{:<20}"*len(headers)
            data_string = "{:<20}{:<20}{:<20.0f}{:<20.0f}{:<20.3f}{:<20.3f}{:<20}"
            if check:
                data_string += "{:<20}"
            print(header_string.format(*headers))
            for line in table:
                print(data_string.format(*line))

        if will_fail:
            raise ValueError("At least one input voltage is out of bounds, with status ERR as seen in the table above")

    def set_current_dac(self, units):
        self.wr_reg("LPGBT.RWF.CUR_DAC.CURDACSELECT", units)

    def get_current_dac(self):
        return self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACSELECT")

    def set_current_dac_uA(self, uA):
        # CURDACSELECT is in units of 900/256 uA per bit, with max of 255
        conv = 256.0/900
        val = min(round(uA*conv), 255)
        self.set_current_dac(val)
        return val/conv

    def get_current_dac_uA(self):
        # CURDACSELECT is in units of 900/256 uA per bit, with max of 255
        return self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACSELECT") * 900/256.0


    def read_adc_raw (self, channel):

        self.kcu.toggle_dispatch()
        self.wr_reg("LPGBT.RW.ADC.ADCINPSELECT", channel)
        self.wr_reg("LPGBT.RW.ADC.ADCINNSELECT", 0xf)

        self.wr_reg("LPGBT.RW.ADC.ADCCONVERT", 0x1)
        self.wr_reg("LPGBT.RW.ADC.ADCENABLE", 0x1)
        self.kcu.dispatch()

        done = 0
        while (done==0):
            #print ("Waiting")
            done = self.rd_reg("LPGBT.RO.ADC.ADCDONE")

        val = self.rd_reg("LPGBT.RO.ADC.ADCVALUEL")
        val |= self.rd_reg("LPGBT.RO.ADC.ADCVALUEH") << 8

        self.kcu.toggle_dispatch()
        self.wr_reg("LPGBT.RW.ADC.ADCCONVERT", 0x0)
        self.wr_reg("LPGBT.RW.ADC.ADCENABLE", 0x1)
        self.kcu.dispatch()

        return val

    def apply_adc_calibration(self, val):
        return val*self.cal_gain/1.85 + (512 - self.cal_offset) # calibrate

    def read_adc(self, channel, calibrate=True, convert=False):

        """
        Reads an ADC channel with optional calibration and conversion

        ADCInPSelect[3:0]  |  Input
        ------------------ |----------------------------------------
        4'd0               |  ADC0 (external pin)
        4'd1               |  ADC1 (external pin)
        4'd2               |  ADC2 (external pin)
        4'd3               |  ADC3 (external pin)
        4'd4               |  ADC4 (external pin)
        4'd5               |  ADC5 (external pin)
        4'd6               |  ADC6 (external pin)
        4'd7               |  ADC7 (external pin)
        4'd8               |  EOM DAC (internal signal)
        4'd9               |  VDDIO * 0.42 (internal signal)
        4'd10              |  VDDTX * 0.42 (internal signal)
        4'd11              |  VDDRX * 0.42 (internal signal)
        4'd12              |  VDD * 0.42 (internal signal)
        4'd13              |  VDDA * 0.42 (internal signal)
        4'd14              |  Temperature sensor (internal signal)
        4'd15              |  VREF/2 (internal signal)
        """

        val=self.read_adc_raw(channel)

        if calibrate:
            val = self.apply_adc_calibration(val)

        if convert:
            conversion = None
            for k in self.adc_mapping.keys():
                if int(self.adc_mapping[k]['pin']) == channel:
                    conversion = self.adc_mapping[k]['conv']
                    break
            if conversion is not None:
                val = val * conversion / (2**10 - 1)
            else:
                raise Exception(f"ADC conversion not found when reading ADC {channel}")

        return val

    def calibrate_adc(self, recalibrate=False):

        def serial_valid(serial):
            return serial != 0

        if (self.ver==0):
            serial = str(self.get_chip_userid())
        else:
            serial = str(self.get_chip_serial())

        cal_file = "lpgbt_adc_calibrations.json"

        # if the json file exists, load it
        if os.path.isfile(cal_file):
            with open(cal_file, 'r') as openfile:
                cal_data = json.load(openfile)
        else:
            cal_data = {}

        # if the serial number is valid and calibration data is stored, just load it from the json
        if serial_valid(serial) and serial in cal_data and not recalibrate:
            gain = cal_data[serial]['gain']
            offset = cal_data[serial]['offset']
            if self.verbose:
                print("Loaded ADC calibration data for chip %s. Gain: %f / Offset: %d" % (serial, gain, offset))

        # else, determine calibration constants
        else:
            print("Recalibrating")
            sleep(0.5)
            # determine offset; both at Vref/2
            offset = self.read_adc_raw(0xf)

            # determine gain; one at Vref/2, one at ground
            # use internal grounding - ADC12, supply voltage divider off
            initial_val = self.rd_reg("LPGBT.RW.ADC.VDDMONENA")
            self.wr_reg("LPGBT.RW.ADC.VDDMONENA", 0x0)

            # ADC = (Vdiff/Vref)*Gain*512 + Offset
            gain = 2*abs(self.read_adc_raw(0xC)-offset)/512
            self.wr_reg("LPGBT.RW.ADC.VDDMONENA", initial_val)
            type = "Trigger" if self.trigger else "DAQ"
            print("Calibrated %s ADC. Gain: %f / Offset: %d" % (type, gain, offset))
            print("Chip %s"%serial)

            if gain < 1.65 or gain > 2 or offset < 490 or offset > 530:
                raise RuntimeError("ADC Calibration Failed!")

            # update and save to json file
            if serial_valid(serial):
                cal_data[serial] = {'gain': gain, 'offset': offset}
                with open(cal_file, "w") as outfile:
                    json.dump(cal_data, outfile)
                    print("Calibration data saved to %s"%cal_file)

        self.cal_gain = gain
        self.cal_offset = offset
        self.calibrated = True

    def load_calibration(self, fin="configs/lpgbt_calibration_latest.zip"):
        # you can also download from https://lpgbt.web.cern.ch/lpgbt/calibration/lpgbt_calibration_latest.zip
        # directly. slow, but works
        import pandas as pd
        if fin.count("https"):
            print("Downloading latest calibration file. This works, but is slow.")
        calib = pd.read_csv(fin, header=3)

        if not hasattr(self, 'chip_serial'):
            self.get_chip_serial()
        if self.chip_serial in calib[['CHIPID']].values:
            return calib[calib.CHIPID==self.chip_serial]
        else:
            print(f"Couldn't find the central calibration for lpGBT chip {self.chip_serial}. Is it from a pre-production series?")
            return None

    def get_current_dac_status(self, channel=0, summary=False):
        enabled = self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.CURDACENABLE")
        currently_set = self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE")
        if summary:
            print("Current source enabled:", enabled==1)
            print(f"- Current: {self.get_current_dac_uA()} uA")
            for i in range(8):
                stat = 'enabled' if (1 << i) & currently_set else 'disabled'
                print(f"- Channel {i}: {stat}")
        return (1 << channel) & currently_set

    def enable_current_source(self):
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.CURDACENABLE", 1)

    def disable_current_source(self):
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.CURDACENABLE", 0)

    def set_current_adc(self, channel, current=100, to=1):
        assert channel < 8, f"Can only choose from ADC0 to ADC7; ADC{channel} was given instead"

        if self.verbose:
            print("Enable ADC current source, current status:", self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.CURDACENABLE"))
        self.enable_current_source()

        currently_set = self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE")

        if self.verbose:
            for i in range(8):
                if (1 << i) & currently_set:
                    print(f"Current source enabled for channel {i}")

        if to==1:
            self.wr_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE", (1 << channel) | currently_set) # Set pin ADC channel to current source
            if self.verbose:
                print(f"Enable current source in pin ADC{channel}...", bin(self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE")))
        else:
            self.wr_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE", ~(1 << channel) & currently_set) # Set pin ADC channel to current source
            if self.verbose:
                print(f"Disable current source in pin ADC{channel}...", bin(self.rd_reg("LPGBT.RWF.CUR_DAC.CURDACCHNENABLE")))

        if self.verbose:
            print(f"Set current source value to {current} uA")
        self.set_current_dac_uA(current)

    def set_dac(self, v_out):
        if v_out > 1.00:
            print ("Can't set the DAC to a value larger than 1.0 V!")
            return
        v_ref = 1.00
        value = min(int(v_out/v_ref*4095), 4095)
        lo_bits = value & 0xFF
        hi_bits = (value & ~lo_bits) >> 8
        self.wr_reg("LPGBT.RWF.CALIBRATION.VREFENABLE", 0x1)  # vref enable
        self.wr_reg("LPGBT.RWF.CALIBRATION.VREFTUNE", 0x63)
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACENABLE", 0x1)
        if self.ver == 0:
            self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEL", lo_bits)
            self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEH", hi_bits)
        elif self.ver == 1:
            self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUE_0TO7", lo_bits)
            self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUE_8TO11", hi_bits)

    def read_dac(self):
        v_ref = 1.00
        if self.ver == 0:
            lo_bits = self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEL")
            hi_bits = self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEH")
        elif self.ver == 1:
            lo_bits = self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUE_0TO7")
            hi_bits = self.rd_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUE_8TO11")
        else:
            raise Exception("Invalid lpgbt version detected.")
        value = lo_bits | (hi_bits << 8)
        return value/4096*v_ref

    def reset_dac(self):
        # reset means: output is set to maximum voltage
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEL", 0x0)
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACVALUEH", 0x0)
        self.wr_reg("LPGBT.RWF.VOLTAGE_DAC.VOLDACENABLE", 0x0)
        self.wr_reg("LPGBT.RWF.CALIBRATION.VREFENABLE", 0x0)

    def initialize(self):
        #if self.trigger and self.rb==0:
        #    self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x0)  # this is already done for v1
        #else:
        #    self.wr_reg("LPGBT.RWF.CHIPCONFIG.HIGHSPEEDDATAOUTINVERT", 0x1)  # this is already done for v1

        # turn on clock outputs
        if (self.verbose):
            print ("Configuring clocks now.")
        self.configure_clocks(0x0fffffff)

        # setup up sca eptx/rx
        # sca_setup() # maybe not needed???

    def loopback(self, nloops=100):
        for i in range(nloops):
            wr = random.randint(0, 255)
            id = "LPGBT.RWF.CHIPID.CHIPID1"
            self.wr_reg(id, wr)
            rd = self.rd_reg(id)
            if wr != rd:
                print("ERR: %d wr=0x%08X rd=0x%08X" % (i, wr, rd))
                return
            if (i % (nloops/100) == 0 and i != 0):
                print("%i reads done..." % i)

    def gpio_init(self, outputs=0x2401):
        self.wr_reg("LPGBT.RWF.PIO.PIODIRH", outputs >> 8)
        self.wr_reg("LPGBT.RWF.PIO.PIODIRL", outputs & 0xFF)

        self.set_gpio(0,1) # GBT_RESET_B
        self.set_gpio(10,1) # VTRX RESET_B
        self.set_gpio(13,0) # VTRX DIS

#    def set_gpio(self, ch, val):
#        if (ch > 7):
#            node = "LPGBT.RWF.PIO.PIOOUTH"
#            ch = ch - 8
#        else:
#            node = "LPGBT.RWF.PIO.PIOOUTL"
#
#        reg = self.get_node(node)
#        adr = reg.address
#        rd = self.rd_adr(adr)
#        if val == 0:
#            rd = rd & (0xff ^ (1 << ch))
#        else:
#            rd = rd | (1 << ch)
#
#        self.wr_adr(adr, rd)

    def reset_pattern_checkers(self):

        self.kcu.action("READOUT_BOARD_%i.LPGBT.PATTERN_CHECKER.RESET" % self.rb)

        for link in (0, 1):
            prbs_en_id = "READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.CHECK_PRBS_EN_%d" % (self.rb, link)
            upcnt_en_id = "READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.CHECK_UPCNT_EN_%d" % (self.rb, link)
            self.kcu.write_node(prbs_en_id, 0)
            self.kcu.write_node(upcnt_en_id, 0)

            self.kcu.write_node(prbs_en_id, 0x00FFFFFF)
            self.kcu.write_node(upcnt_en_id, 0x00FFFFFF)

        self.kcu.action("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.CNT_RESET" % self.rb)

    def read_pattern_checkers(self, quiet=False, log=True, log_dir="./tests/"):
        if log_dir and os.path.isfile(log_dir + "pattern_checks.p") and log:
            log_dict = pickle.load(open(log_dir + "pattern_checks.p", "rb"))
        elif log:
            default_dict = {key:{"error":[], "total_frames":[]} for key in range(24)}
            link_dict = {"PRBS":copy.deepcopy(default_dict), "UPCNT":copy.deepcopy(default_dict)}
            log_dict = {"Link 0":copy.deepcopy(link_dict), "Link 1":copy.deepcopy(link_dict)}

        for link in (0, 1):

            prbs_en = self.kcu.read_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.CHECK_PRBS_EN_%d" % (self.rb, link))
            upcnt_en = self.kcu.read_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.CHECK_UPCNT_EN_%d" % (self.rb, link))

            prbs_errs = 28*[0]
            upcnt_errs = 28*[0]

            for mode in ["PRBS", "UPCNT"]:
                if quiet is False:
                    print("Link " + str(link) + " " + mode + ":")
                for i in range(28):

                    check = False

                    if mode == "UPCNT" and ((upcnt_en >> i) & 0x1):
                        check = True
                    if mode == "PRBS" and ((prbs_en >> i) & 0x1):
                        check = True

                    if check:
                        self.kcu.write_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.SEL" % (self.rb), link*28+i)

                        uptime_msbs = self.kcu.read_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.TIMER_MSBS" % (self.rb))
                        uptime_lsbs = self.kcu.read_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.TIMER_LSBS" % (self.rb))

                        uptime = (uptime_msbs << 32) | uptime_lsbs

                        errs = self.kcu.read_node("READOUT_BOARD_%d.LPGBT.PATTERN_CHECKER.%s_ERRORS" % (self.rb, mode))

                        if quiet is False:
                            s = "    Channel %02d %s bad frames of %s (%.0f Gb)" % (i, ("{:.2e}".format(errs)), "{:.2e}".format(uptime), uptime*8/1000000000.0)
                            if (errs == 0):
                                s += " (ber <%s)" % ("{:.1e}".format(1/(uptime*8)))
                                print(colors.green(s))
                            else:
                                s += " (ber>=%s)" % ("{:.1e}".format((1.0*errs)/uptime))
                                print(colors.red(s))

                        if mode == "UPCNT":
                            upcnt_errs[i] = errs
                        if mode == "PRBS":
                            prbs_errs[i] = errs
                        if log:
                            log_dict["Link {}".format(link)][mode][i]["error"].append(int(errs))
                            log_dict["Link {}".format(link)][mode][i]["total_frames"].append(int(uptime))

                    else:
                        if mode == "UPCNT":
                            upcnt_errs[i] = 0xFFFFFFFF
                        if mode == "PRBS":
                            prbs_errs[i] = 0xFFFFFFFF
        if log and log_dir:
            pickle.dump(log_dict, open(log_dir + "pattern_checks.p", "wb"))
        return log_dict

    def set_uplink_group_data_source(self, type, pattern=0x55555555):
        setting = 0
        if (type == "normal"):
            setting = 0
        elif(type == "prbs7"):
            setting = 1
        elif(type == "cntup"):
            setting = 2
        elif(type == "cntdown"):
            setting = 3
        elif(type == "pattern"):
            setting = 4
        elif(type == "invpattern"):
            setting = 5
        elif(type == "loopback"):
            setting = 6
        else:
            print("Setting invalid in set_uplink_group_data_source")
            return

        for i in range(7):
            self.wr_reg("LPGBT.RW.TESTING.ULG%dDATASOURCE"%i, setting)

        if (setting == 4 or setting == 5):
            for i in range(4):
                self.wr_reg("LPGBT.RW.TESTING.DPDATAPATTERN%d"%i, 0xff&(pattern >> (i*8)))

    def set_downlink_data_src(self, source):
        id = "READOUT_BOARD_%d.LPGBT.DOWNLINK.DL_SRC" % self.rb
        if (source == "etroc"):
            self.kcu.write_node(id, 0)
        if (source == "upcnt"):
            self.kcu.write_node(id, 1)
        if (source == "prbs"):
            self.kcu.write_node(id, 2)

    def set_ps0_phase(self, phase):
        phase = phase & 0x1ff
        msb = 0x1 & (phase >> 8)
        self.wr_reg("LPGBT.RWF.PHASE_SHIFTER.PS0ENABLEFINETUNE", 1)
        self.wr_reg("LPGBT.RWF.PHASE_SHIFTER.PS0DELAY_7TO0", 0xff & phase)
        self.wr_reg("LPGBT.RWF.PHASE_SHIFTER.PS0DELAY_8", msb)

    def I2C_write_single(self, reg=0x0, val=0, master=2, slave_addr=0x70, freq=2):
        pass

    def I2C_write(self, reg=0x0, val=10, master=2, slave_addr=0x70, adr_nbytes=2, freq=2, verbose=False, ignore_response=False):
        '''
        reg: target register
        val: has to be a single byte, or a list of single bytes.
        master: lpGBT master (2 by default)
        this function is following https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#example-2-multi-byte-write
        '''

        self.kcu.toggle_dispatch()

        i2cm = master

        i2cm1cmd = self.get_node('LPGBT.RW.I2C.I2CM1CMD').real_address
        i2cm0cmd = self.get_node('LPGBT.RW.I2C.I2CM0CMD').real_address

        # god fucking damnit
        if self.ver == 0:
            i2cm1status = self.LPGBT_CONST.I2CM1STATUS
            i2cm0status = self.LPGBT_CONST.I2CM0STATUS
        else:
            i2cm1status = self.get_node('LPGBT.RO.I2CREAD.I2CM1STATUS').real_address
            i2cm0status = self.get_node('LPGBT.RO.I2CREAD.I2CM0STATUS').real_address

        i2cm0data0 = self.get_node('LPGBT.RW.I2C.I2CM0DATA0').real_address
        i2cm0cmd = self.get_node('LPGBT.RW.I2C.I2CM0CMD').real_address
        i2cm0address = self.get_node('LPGBT.RW.I2C.I2CM0ADDRESS').real_address

        OFFSET_WR = i2cm*(i2cm1cmd - i2cm0cmd) #using the offset trick to switch between masters easily
        OFFSET_RD = i2cm*(i2cm1status - i2cm0status)

        adr_bytes = [ ((reg >> (8*i)) & 0xff) for i in range(adr_nbytes) ]

        if type(val == int):
            data_bytes = [val]
        elif type(val == list):
            data_bytes = val
        else:
            raise RuntimeError("Data must be an int or list of ints")

        nbytes = len(adr_bytes+data_bytes)

        self.wr_adr(
            i2cm0data0+OFFSET_WR,
            nbytes<<self.LPGBT_CONST.I2CM_CR_NBYTES_of | freq<<self.LPGBT_CONST.I2CM_CR_FREQ_of,
        )
        self.wr_adr(
            i2cm0cmd+OFFSET_WR,
            self.LPGBT_CONST.I2CM_WRITE_CRA,
        )

        for i, data_byte in enumerate(adr_bytes+data_bytes):
            page    = int(i/4)
            offset  = int(i%4)

            self.wr_adr(
                i2cm0data0 + OFFSET_WR + offset,
                data_byte
            )

            if i%4==3 or i==(nbytes-1):
                self.wr_adr(
                    i2cm0cmd+OFFSET_WR,
                    self.LPGBT_CONST.I2CM_W_MULTI_4BYTE0+page,
                )

        self.wr_adr(i2cm0address+OFFSET_WR, slave_addr)# write the address of the follower
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_WRITE_MULTI)# execute write (c)

        self.kcu.dispatch()

        if not ignore_response:
            status = self.rd_adr(i2cm0status+OFFSET_RD)
            retries = 0
            while (status != self.LPGBT_CONST.I2CM_SR_SUCC_bm):
                status = self.rd_adr(i2cm0status+OFFSET_RD).value()
                retries += 1
                if retries > 50:
                    raise TimeoutError(f"I2C write failed after 50 retries, status={status}")

    def I2C_read(self, reg=0x0, master=2, slave_addr=0x70, nbytes=1, adr_nbytes=2, freq=2, verbose=False, timeout=0.1):
        #https://gitlab.cern.ch/lpgbt/pigbt/-/blob/master/backend/apiapp/lpgbtLib/lowLevelDrivers/MASTERI2C.py#L83

        # debugging
        #print("### LPGBT.I2C_read ###")
        #print(f"reg: {reg}, \tmaster: {master}, \tslave_addr: {slave_addr}, \tnbytes: {nbytes}, \tadr_nbytes: {adr_nbytes}, \tfreq: {freq}, \tver: {self.ver}")

        i2cm      = master

        i2cm1cmd = self.get_node('LPGBT.RW.I2C.I2CM1CMD').real_address
        i2cm0cmd = self.get_node('LPGBT.RW.I2C.I2CM0CMD').real_address

        # debugging
        #print(f"i2cm1cmd: {i2cm1cmd}, \ti2cm0cmd: {i2cm0cmd}")

        if self.ver == 0:
            i2cm1status = self.LPGBT_CONST.I2CM1STATUS
            i2cm0status = self.LPGBT_CONST.I2CM0STATUS
        else:
            i2cm1status = self.get_node('LPGBT.RO.I2CREAD.I2CM1STATUS').real_address
            i2cm0status = self.get_node('LPGBT.RO.I2CREAD.I2CM0STATUS').real_address

        i2cm0data0 = self.get_node('LPGBT.RW.I2C.I2CM0DATA0').real_address
        i2cm0cmd = self.get_node('LPGBT.RW.I2C.I2CM0CMD').real_address
        i2cm0address = self.get_node('LPGBT.RW.I2C.I2CM0ADDRESS').real_address

        OFFSET_WR = i2cm*(i2cm1cmd - i2cm0cmd) #using the offset trick to switch between masters easily
        OFFSET_RD = i2cm*(i2cm1status - i2cm0status)

        # debugging
        #print(f"i2cm1status: {i2cm1status}, \ti2cm0status: {i2cm0status}, \ti2cm0data0: {i2cm0data0}, \ti2cm0cmd: {i2cm0cmd}, \ti2cm0address: {i2cm0address}, \tOFFSET_WR: {OFFSET_WR}, \tOFFSET_RD: {OFFSET_RD}")

        ################################################################################
        # Write the register address
        ################################################################################

        self.kcu.toggle_dispatch()

        # https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#i2c-write-cr-0x0
        self.wr_adr(i2cm0data0+OFFSET_WR, adr_nbytes<<self.LPGBT_CONST.I2CM_CR_NBYTES_of | (freq<<self.LPGBT_CONST.I2CM_CR_FREQ_of))
        # debugging
        #print(f"Address: {i2cm0data0+OFFSET_WR}, \tValue: {adr_nbytes<<self.LPGBT_CONST.I2CM_CR_NBYTES_of | (freq<<self.LPGBT_CONST.I2CM_CR_FREQ_of)}")
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_WRITE_CRA) #write to config register
        # debugging
        #print(f"Address: {i2cm0cmd+OFFSET_WR}, \tValue: {self.LPGBT_CONST.I2CM_WRITE_CRA}")

        # https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#i2c-w-multi-4byte0-0x8
        for i in range (adr_nbytes):
            self.wr_adr(self.get_node("LPGBT.RW.I2C.I2CM0DATA%d"%i).real_address + OFFSET_WR, (reg >> (8*i)) & 0xff )
            # debugging
            #print(f"Address: {self.get_node('LPGBT.RW.I2C.I2CM0DATA%d'%i).real_address + OFFSET_WR}, \tValue: {(reg >> (8*i)) & 0xff}, \ti: {i}")
        # self.wr_adr(self.LPGBT_CONST.I2CM0DATA1 + OFFSET_WR , regh)
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_W_MULTI_4BYTE0) # prepare a multi-write
        # debugging
        #print(f"Address: {i2cm0cmd+OFFSET_WR}, \tValue: {self.LPGBT_CONST.I2CM_W_MULTI_4BYTE0}")

        # https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#i2c-write-multi-0xc
        self.wr_adr(i2cm0address+OFFSET_WR, slave_addr)
        # debugging
        #print(f"Address: {i2cm0address+OFFSET_WR}, \tValue: {slave_addr}")
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_WRITE_MULTI)# execute multi-write
        # debugging
        #print(f"Address: {i2cm0cmd+OFFSET_WR}, \tValue: {self.LPGBT_CONST.I2CM_WRITE_MULTI}")

        self.kcu.dispatch()

        status = self.rd_adr(i2cm0status+OFFSET_RD)

        # debugging
        #print(f"status: {status}, LPGBT_CONST.I2CM_SR_SUCC_bm: {self.LPGBT_CONST.I2CM_SR_SUCC_bm}, Address: {i2cm0status+OFFSET_RD}")

        retries = 0
        while (status != self.LPGBT_CONST.I2CM_SR_SUCC_bm):
            status = self.rd_adr(i2cm0status+OFFSET_RD).value()
            # debugging
            #print(f"Updating status: {status}, retries: {retries}")
            retries += 1
            if retries > 50:
                raise TimeoutError(f"I2C transaction failed after 50 retries because of an issue in writing the register address, status={status}")

        ################################################################################
        # Write the data
        ################################################################################

        self.kcu.toggle_dispatch()

        # https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#i2c-write-cr-0x0
        self.wr_adr(i2cm0data0+OFFSET_WR, nbytes<<self.LPGBT_CONST.I2CM_CR_NBYTES_of | freq<<self.LPGBT_CONST.I2CM_CR_FREQ_of)
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_WRITE_CRA) #write to config register

        # https://lpgbt.web.cern.ch/lpgbt/v0/i2cMasters.html#i2c-read-multi-0xd
        self.wr_adr(i2cm0address+OFFSET_WR, slave_addr) #write the address of follower first
        self.wr_adr(i2cm0cmd+OFFSET_WR, self.LPGBT_CONST.I2CM_READ_MULTI)# execute read

        self.kcu.dispatch()

        status = self.rd_adr(i2cm0status+OFFSET_RD)

        # debugging
        #print(f"status: {status}")

        retries = 0
        while (status != self.LPGBT_CONST.I2CM_SR_SUCC_bm):
            status = self.rd_adr(i2cm0status+OFFSET_RD).value()
            retries += 1
            if retries > 50:
                raise TimeoutError(f"I2C transaction failed after 50 retries because of an issue in reading back the data, status={status}")

        read_values = []

        if self.ver == 0:
            i2cm0read15 = self.LPGBT_CONST.I2CM0READ15
        else:
            i2cm0read15 = self.get_node("LPGBT.RO.I2CREAD.I2CM0READ.I2CM0READ15").real_address

        # debugging
        #print(f"i2cm0read15: {i2cm0read15}")

        for i in range(0, nbytes):
            tmp_adr = abs(i-i2cm0read15)+OFFSET_RD
            # debugging
            #print(f"tmp_adr: {tmp_adr}, \ttmp_adr_val: {self.rd_adr(tmp_adr).value()}")
            read_values.append(self.rd_adr(tmp_adr).value())

        #read_value = self.rd_adr(self.LPGBT_CONST.I2CM0READ15+OFFSET_RD) # get the read value. this is just the first byte
        if nbytes==1:
            return read_values[0]  # this is so bad, but needed for compatibility with wr_reg
        else:
            return read_values

    def program_slave_from_file (self, filename, master=2, slave_addr=0x70):
        if self.verbose:
            print(" > Programming Trigger lpGBT from file.")
        f = open(filename, "r")
        for line in f:
            adr, data = line.split(" ")
            adr = int(adr)
            wr = int(data.replace("0x",""), 16)
            if (wr != 0):
                if self.verbose:
                    print("Writing address: %d, value: 0x%02x" % (adr, wr))
                self.I2C_write(reg=adr, val=wr, master=master, slave_addr=slave_addr, ignore_response=True)
                #rd = self.I2C_read(reg=adr, master=master, slave_addr=slave_addr)
                #if (wr!=rd):
                #    print("LPGBT readback error 0x%02X != 0x%02X at adr %d" % (wr, rd, adr))
        self.kcu.dispatch()

    def read_temp_i2c(self):
        res = self.I2C_read(reg=0x0, master=1, slave_addr=0x48, nbytes=2)
        temp_dig = (res[0] << 4) + (res[1] >> 4)
        return temp_dig*0.0625

    def get_board_id(self):
        '''
        |-------+------+---------------+--------------------------------------|
        | Range | Bits | Meaning       | Description                          |
        |-------+------+---------------+--------------------------------------|
        |  15:0 |   16 | Serial Number | Board serial number                  |
        | 31:29 |    3 | Version Major | Major version of RB (e.g. 1 in v1.6) |
        | 28:25 |    4 | Version Minor | Major version of RB (e.g. 6 in v1.6) |
        | 24:23 |    2 | LPGBT Version | 0x0 = v0; 0x1 = v1                   |
        | 22:19 |    4 | Board Flavor  | 0x0 = 3 module; 0x1 =                |
        |-------+------+---------------+--------------------------------------|

        * User IDs
        #+begin_src python :results output
        def user_id (serial, major, minor, lpgbt, flavor):
          data = 0
          data |= serial & 0xffff
          data |= (major & 0x7) << 29
          data |= (minor & 0xf) << 25
          data |= (lpgbt & 0x3) << 23
          data |= (flavor & 0xf) << 19
          return data

        0x003 -> 7:0
        0x002 -> 15:8
        0x001 -> 23:16
        0x000 ->

        '''

        board_id = {}
        n_module = {0:3, 1:6, 2:7}
        flavors = {0: '3 module', 1: '6 module', 2: '7 module'}

        user_id =   self.get_chip_userid()
        board_id['rb_ver_major']    = user_id >> 29
        board_id['rb_ver_minor']    = user_id >> 25 & (2**4-1)
        board_id['lpgbt_ver']       = user_id >> 23 & (2**2-1)
        board_id['rb_flavor']       = flavors[user_id >> 19 & (2**4-1)]
        board_id['n_module']        = n_module[user_id >> 19 & (2**4-1)]
        board_id['serial_number']   = user_id & (2**16-1)
        board_id['lpgbt_serial']    = self.get_chip_serial()

        return board_id

    def eyescan(self, end_of_count_sel=7):

        self.wr_reg("LPGBT.RW.EOM.EOMENDOFCOUNTSEL", end_of_count_sel)
        self.wr_reg("LPGBT.RW.EOM.EOMENABLE", 1)

        # Equalizer settings
        self.wr_reg("LPGBT.RWF.EQUALIZER.EQCAP", 0x1)
        self.wr_reg("LPGBT.RWF.EQUALIZER.EQRES0", 0x1)
        self.wr_reg("LPGBT.RWF.EQUALIZER.EQRES1", 0x1)
        self.wr_reg("LPGBT.RWF.EQUALIZER.EQRES2", 0x1)
        self.wr_reg("LPGBT.RWF.EQUALIZER.EQRES3", 0x1)

        datavalregh    = "LPGBT.RO.EOM.EOMCOUNTERVALUEH"
        datavalregl    = "LPGBT.RO.EOM.EOMCOUNTERVALUEL"
        eomphaseselreg = "LPGBT.RW.EOM.EOMPHASESEL"
        eomstartreg    = "LPGBT.RW.EOM.EOMSTART"
        eomstatereg    = "LPGBT.RO.EOM.EOMSMSTATE"
        eombusyreg     = "LPGBT.RO.EOM.EOMBUSY"
        eomendreg      = "LPGBT.RO.EOM.EOMEND"
        eomvofsel      = "LPGBT.RW.EOM.EOMVOFSEL"

        cntvalmax = 0
        cntvalmin = 2**20

        ymin=0
        ymax=30
        xmin=0
        xmax=64

        eyeimage = [[0 for x in range(xmin, xmax)] for y in range(ymin, ymax)]

        print("\nRunning eye scan...")
        for y_axis in range(ymin, ymax):

            # update yaxis
            self.wr_reg(eomvofsel, y_axis)

            for x_axis in range(xmin, xmax):

                x_axis_wr = x_axis

                # update xaxis
                self.wr_reg(eomphaseselreg, x_axis_wr)

                # start measurement
                self.wr_reg(eomstartreg, 0x1)

                countervalue = (self.rd_reg(datavalregh)) << 8 | self.rd_reg(datavalregl)
                if (countervalue > cntvalmax):
                    cntvalmax = countervalue
                if (countervalue < cntvalmin):
                    cntvalmin = countervalue
                eyeimage[y_axis][x_axis] = countervalue

                # deassert eomstart bit
                self.wr_reg(eomstartreg, 0x0)

        print("Counter value max=%d\n" % cntvalmax)

        # normalize for plotting and save to file
        normalize = lambda val : int(100*(cntvalmax - val)/(cntvalmax-cntvalmin))
        eye_scan_data = [[normalize(x) for x in y] for y in eyeimage]

        if not os.path.isdir("eye_scan_results"):
            os.mkdir("eye_scan_results")

        serialnums = "lpgbt%s_kcu%s" %(self.get_chip_serial(), self.kcu.get_serial())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "%s_%s" %(serialnums, timestamp)

        with open("eye_scan_results/%s.json" %filename, "w") as outfile:
            json.dump(eye_scan_data, outfile)
        print("Data saved to eye_scan_results/%s.json\n" %filename)


        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)
        (fig, axs) = plt.subplots(1, 1, figsize=(10, 8))
        print ("fig type = " + str(type(fig)))
        print ("axs type = " + str(type(axs)))
        axs.set_title("LpGBT 2.56 Gbps RX Eye Opening Monitor")
        plot = axs.imshow(eye_scan_data, alpha=0.9, vmin=0, vmax=100, cmap='jet',interpolation="nearest", aspect="auto",extent=[-384.52/2,384.52/2,-0.6,0.6,])
        plt.xlabel('ps')
        plt.ylabel('volts')
        fig.colorbar(plot, ax=axs)

        #plt.show()
        fig.savefig(f'eye_scan_results/{filename}.png')
        print("Eye diagram saved to eye_scan_results/%s.json\n" %filename)

        # print results to bash
        try:
            from colored import fg, bg, attr
            color_scale = [124,196,202,214,190,82,50,38,21,19]
            sys.stdout.write("Color Scale: ")
            for i in range(10):
                sys.stdout.write("%s%01d%s" % (bg(color_scale[i]),i,attr('reset')))
            sys.stdout.write("\n\n")

            for y_axis in range(ymin, ymax):
                for x_axis in range(xmin, xmax):
                    printval = int(eyeimage[y_axis][x_axis]/1000)
                    sys.stdout.write("%s%01d%s" % (bg(color_scale[printval]), printval, attr('reset')))
                    sys.stdout.flush()
                sys.stdout.write("\n")
        except:
            print("Need to pip install colored to print out results.")
            print("Eye scan results were still saved and can be plotted.")


    def get_chip_userid(self):
        return self.rd_reg("LPGBT.RWF.CHIPID.USERID3") << 24 |\
               self.rd_reg("LPGBT.RWF.CHIPID.USERID2") << 16 |\
               self.rd_reg("LPGBT.RWF.CHIPID.USERID1") << 8 |\
               self.rd_reg("LPGBT.RWF.CHIPID.USERID0")

    def get_chip_id(self):
        return self.rd_reg("LPGBT.RWF.CHIPID.CHIPID3") << 24 |\
               self.rd_reg("LPGBT.RWF.CHIPID.CHIPID2") << 16 |\
               self.rd_reg("LPGBT.RWF.CHIPID.CHIPID1") << 8 |\
               self.rd_reg("LPGBT.RWF.CHIPID.CHIPID0")

    def get_chip_serial(self):
        if self.ver == 1:
            # NOTE we have to read from the fuses directly.
            # ideally this can still be verified (May 2023)
            self.wr_adr(0x119, 0x1 << 1)  # write FuseRead https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#reg-fusecontrol
            while True:
                # wait for FuseDataValid https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#reg-fusestatus
                if self.rd_adr(0x1b1) >> 2 == 1: break

            chipids = []
            # there should be 5 copies of the chipid, but I can only find 4
            # there's nothing else in the fuses that's non-zero
            for i in range(4):
                self.wr_adr(0x11f, i)
                #chipids_all.append((self.rd_adr(0x1b2).value(), self.rd_adr(0x1b3).value(), self.rd_adr(0x1b4).value(), self.rd_adr(0x1b5).value()))
                chipids.append(self.rd_adr(0x1b2) << 0 | self.rd_adr(0x1b3) << 8 | self.rd_adr(0x1b4) << 16 | self.rd_adr(0x1b5) << 24)

            self.wr_adr(0x119, 0)  # write FuseRead https://lpgbt.web.cern.ch/lpgbt/v1/registermap.html#reg-fusecontrol

            #print(chipids_all)


            if all([c==chipids[0] for c in chipids]):
                self.chip_serial = hex(chipids[0]).upper()[2:]
                return self.chip_serial
            else:
                print("CHIPD serial needs majority vote")
                self.chip_serial = hex(majority_vote(chipids, majority=3)).upper()[2:]
                return self.chip_serial

        elif self.ver == 0:
            # NOTE: this is what's supposed to work for lpGBT v0
            # but note sure if that's actually true
            self.chip_serial = self.rd_reg("LPGBT.RWF.CHIPID.CHIPID3") << 24 |\
                self.rd_reg("LPGBT.RWF.CHIPID.CHIPID2") << 16 |\
                self.rd_reg("LPGBT.RWF.CHIPID.CHIPID1") << 8 |\
                self.rd_reg("LPGBT.RWF.CHIPID.CHIPID0")
            return hex(self.chip_serial).upper()[2:]

    def get_power_up_state_machine(self, quiet=True):

        pusmstate = self.rd_reg("LPGBT.RO.PUSM.PUSMSTATE")

        if not quiet:

            print ("PUSM State:")

            if (pusmstate==0):  print ("\t0  = ARESET - the FSM stays in this state when power-on-reset or an external reset (RSTB) is asserted. \n\t When external signal PORdisable is asserted, the signal generated by the internal power-on-reset is ignored. All action flags are reset in this state.")
            if (pusmstate==1):  print ("\t1  = RESET - synchronous reset state. In this state, the FSM produces synchronous reset signal for various circuits. \n\t All action flags are not reset in this state.")
            if (pusmstate==2):  print ("\t2  = WAIT_VDD_STABLE - the FSM waits for VDD to raise. It has fixed duration of 4,000 clock cycles (~100us).")
            if (pusmstate==3):  print ("\t3  = WAIT_VDD_HIGHER_THAN_0V90 - the FSM monitors the VDD voltage. \n\t It waits until VDD stays above 0.9V for a period longer than 1us.\n\t This state is bypassed if PORdisable is active.")
            if (pusmstate==4):  print ("\t4  = FUSE_SAMPLING - initiate fuse sampling.")
            if (pusmstate==5):  print ("\t5  = UPDATE FROM FUSES - transfer fuse values into registers. Transfer executed only if updateEnable fuse in POWERUP2 register is blown.")
            if (pusmstate==6):  print ("\t6  = PAUSE_FOR_PLL_CONFIG - this state is foreseen for initial testing of the chip when optimal registers settings are not yet known and the e-fuses have not been burned. The FSM will wait in this state until pllConfigDone bit is asserted. While in this state, the user can use the I2C interface to write values to the registers. For more details about intended use please refer to Section 3.7.")
            if (pusmstate==7):  print ("\t7  = WAIT_POWER_GOOD - this state is foreseen to make sure that the power supply voltage is stable before proceeding with further initialization. When PGEnable bit is enabled the FSM will wait until VDD level stays above value configured by PGLevel[2:0] for longer than time configured by PGDelay[4:0]. If PGEnable is not set, one can use PGDelay[4:0] as a fixed delay. The PGLevel[2:0] and PGDelay[4:0] are interpreted according to Table 8.1 and Table 8.2.")
            if (pusmstate==8):  print ("\t8  = RESETOUT - in this state a reset signal is generated on the resetout pin. The reset signal is active low. The duration of the reset pulse is controlled by value of ResetOutLength[1:0] field according to Table 8.3.")
            if (pusmstate==9):  print ("\t9  = I2C_TRANS - this state is foreseen to execute one I2C transaction. This feature can be used to configure a laser driver chip or any other component in the system. To enable transaction, the I2CMTransEnable bit has to be programmed and master channel has to be selected by I2CMTransChannel[1:0]. Remaining configuration like I2CMTransAddressExt[2:0], I2CMTransAddress[6:0], and I2CMTransCtrl[127:0] should be configured according to the description in the I2C slaves chapter.")
            if (pusmstate==10): print ("\t10 = RESET_PLL - reset PLL/CDR control logic.")
            if (pusmstate==11): print ("\t11 = WAIT_PLL_LOCK - waits for the PLL/CDR to lock. \n\t When lpGBT is configured in simplex RX or transceiver mode the lock signal comes from frame aligner. \n\t It means that the valid lpGBT frame has to be sent in the downlink. \n\t This state can be interrupted by timeout action (see the description below).")
            if (pusmstate==12): print ("\t12 = INIT_SCRAM - initializes scrambler in the uplink data path.")
            if (pusmstate==13): print ("\t13 = PAUSE_FOR_DLL_CONFIG - this state is foreseen for the case in which user wants to use serial interface (IC/EC) to configure the chip. The FSM will wait in this state until dllConfigDone bit is asserted. While in this state, the user can use the serial interface (IC/EC) or I2C interface to write values to the registers. For more details about intended use please refer to Section 3.7.")
            if (pusmstate==14): print ("\t14 = RESET_DLLS - reset DLLs in ePortRx groups and phase-shifter.")
            if (pusmstate==15): print ("\t15 = WAIT_DLL_LOCK - wait until all DLL report to be locked. This state can be interrupted by timeout action (see the description below).")
            if (pusmstate==16): print ("\t16 = RESET_LOGIC_USING_DLL - reset a logic using DLL circuitry. In case of ePortRx groups, this signal is used to initialize automatic phase training. This state has no impact on a phase-shifter operation.")
            if (pusmstate==17): print ("\t17 = WAIT_CHNS_LOCKED - in this state, FSM waits until automatic phase training is finished for all enabled ePortRx groups. One should keep in mind, that data transitions have to be present on the enabled channels to acquire lock. By default this state is bypassed, it can be enabled asserting PUSMReadyWhenChnsLocked bit in POWERUP register. This state can be interrupted by timeout action (see the description below).")
            if (pusmstate==18): print ("\t18 = READY - initialization is completed. Chip is operational. READY signal is asserted.")

        return pusmstate

    def monitor_pusm(self, maxcount=10000):
        print ("Initial PUSM state:")
        tmp = self.get_power_up_state_machine(quiet=False)
        for i in range(maxcount):
            pusm = self.get_power_up_state_machine()
            if not (tmp==pusm):
                print ("Changed state to:", pusm)
            tmp = pusm

    def dump_config(self, out_file=None):
        #config = ''
        max_adr = 0x13c if self.ver ==0 else 0x14c  # last registers for v0 and v1, including debug registers
        for i in range(max_adr+1):
            #config += f"{i} {self.rd_adr(i)}\n"
            print (f"{i} {self.rd_adr(i)}")

        if out_file:
            with open(out_file, 'w') as f:
                for i in range(max_adr+1):
                    f.write(f"{i} {hex(self.rd_adr(i))}\n")

    def set_configured(self):
        self.wr_reg("LPGBT.RWF.CHIPID.USERID0", 0xFF)
        return self.is_configured()

    def is_configured(self):
        return self.rd_reg("LPGBT.RWF.CHIPID.USERID0") == 0xFF

    def set_power_up_done(self):
        self.wr_reg("LPGBT.RWF.CHIPID.USERID1", 0xAA)
        return self.power_up_done()

    def power_up_done(self):
        return self.rd_reg("LPGBT.RWF.CHIPID.USERID1") == 0xAA

if __name__ == '__main__':

    lpgbt = LPGBT()
    lpgbt.get_version()
    lpgbt.parse_xml(self.ver)
    lpgbt.dump(nMax=10)
