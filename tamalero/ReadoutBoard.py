import os
from tamalero.LPGBT import LPGBT
from tamalero.SCA import SCA
# from tamalero.MUX64 import MUX64
from tamalero.utils import get_temp, chunk, get_temp_direct, get_config, load_yaml
from tamalero.VTRX import VTRX
from tamalero.utils import read_mapping
from tamalero.colors import red, green
import time, datetime, json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tamalero.Module import Module

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    print ("Package `tabulate` not found.")

from time import sleep

flavors = {
    'small': 3,
    'medium': 6,
    'large': 7,
}

class ReadoutBoard:

    def __init__(self, rb=0, trigger=True, flavor='small', kcu=None, config='default', alignment=False, data_mode=True, etroc='ETROC2', verbose=False, allow_bad_links=False, poke=False):
        '''
        create a readout board.
        trigger: if true, also configure a trigger lpGBT
        '''
        # print('############################version check############################')
        self.rb = rb
        self.flavor = flavor
        self.ver = 2
        self.nmodules = flavors[flavor]
        self.config = config

        self.trigger = trigger
        # print('############################ReadoutBoard LPGBT Instantiation############################')
        self.DAQ_LPGBT = LPGBT(rb=rb, flavor=flavor, kcu=kcu, config=self.config, poke=poke, rbver=self.ver)
        # print('############################ReadoutBoard VTRX Instantiation############################')
        self.VTRX = VTRX(self.DAQ_LPGBT)
        # This is not yet recommended:
        #for adr in [0x06, 0x0A, 0x0E, 0x12]:
        #    self.VTRX.wr_adr(adr, 0x20)

        self.alignment = alignment
        self.data_mode = data_mode
        self.etroc = etroc
        self.verbose = verbose

        if kcu != None:
            self.kcu = kcu
            self.kcu.readout_boards.append(self)
            self.DAQ_LPGBT.configure()
            # If version is undetermined or older than 3, get version from LPGBT and try to connect SCA to KCU
            # self.SCA = SCA(rb=rb, flavor=flavor, ver=self.DAQ_LPGBT.ver, config=self.config, poke=poke)
            if self.DAQ_LPGBT.ver == 1:
                self.ver = 2
                # self.SCA.update_ver(self.ver)
                self.DAQ_LPGBT.update_rb_ver(self.ver)
            elif self.DAQ_LPGBT.ver == 0:
                self.ver = 1
                # self.SCA.update_ver(self.ver)
                self.DAQ_LPGBT.update_rb_ver(self.ver)
            # self.SCA.connect_KCU(kcu)
            try:
                self.sca_hard_reset()
                self.sca_setup(verbose=self.verbose)
                # self.SCA.reset()
                # self.SCA.connect()
                # self.SCA.configure_control_registers()
                # self.SCA.config_gpios()  # this sets the directions etc according to the mapping
                # if self.verbose:
                    # print(" > GBT-SCA detected and configured")
            except TimeoutError:
                self.ver = 3
                self.DAQ_LPGBT.update_rb_ver(self.ver)
            # If version newer than 3, connect MUX64
            # if self.ver > 2:
            #     self.MUX64 = MUX64(rb=self.rb, ver=1, config=self.config, rbver=self.ver, LPGBT=self.DAQ_LPGBT)

        if self.verbose:
            print(f" > Readout Board version detected: {self.ver}")

        self.configuration = get_config(self.config, version=f'v{self.ver}')

        if poke:
            self.get_trigger(poke=True)
            return

        self.enable_etroc_readout()  # enable readout of all ETROCs by default
        self.enable_etroc_readout(slave=True)  # enable readout of all ETROCs by default

        self.configured = self.DAQ_LPGBT.is_configured()
        #if not self.configured:

        # self.VTRX.get_version()

        if trigger:
            self.get_trigger()

            if self.trigger:
                if not self.TRIG_LPGBT.power_up_done():
                    self.TRIG_LPGBT.power_up_init()

                self.TRIG_LPGBT.invert_links()

        # if not self.configured:
        #     self.configure()
        #     self.configured = 1

        if self.ver == 2:
            # this method does not work for RB v1 / lpGBT v0
            self.reset_problematic_links(max_retries=50, allow_bad_links=allow_bad_links)

        # print('############################ReadoutBoard self.is_configured Debug############################')
        self.is_configured()


    def get_trigger(self, poke=False):
        # Self-check if a trigger lpGBT is present, if trigger is not explicitely set to False
        sleep(0.5)
        try:
            test_read = self.DAQ_LPGBT.I2C_read(reg=0x0, master=2, slave_addr=0x70, verbose=False)
        except TimeoutError:
            test_read = None
        if test_read is not None and self.trigger and not poke:
            print (" > Found trigger lpGBT, will configure it now.")
            self.trigger = True
            print (" > Enabling VTRX channel for trigger lpGBT")
            self.VTRX.enable(ch=1)
            sleep(1)
        elif test_read is None:
            print ("No trigger lpGBT found.")
            self.trigger = False
        else:
            if self.verbose:
                print ("Trigger lpGBT was found, but will not be configured.")

        if self.trigger:
            self.TRIG_LPGBT = LPGBT(rb=self.rb, flavor=self.flavor, trigger=True, master=self.DAQ_LPGBT, kcu=self.kcu, config=self.config, poke=poke, rbver=self.ver)


    def connect_KCU(self, kcu):
        self.kcu = kcu
        self.DAQ_LPGBT.connect_KCU(kcu)
        # self.SCA.connect_KCU(kcu)

    def set_elink_width(self, width=320):
        widths = {320:1, 640:2, 1280:3}
        self.kcu.write_node("READOUT_BOARD_%d.ELINK_WIDTH"%self.rb, widths[width]+1)
        for i in range(7):
            # set banks to 320 Mbps (1) or 640 Mbps (2)
            self.DAQ_LPGBT.wr_reg("LPGBT.RWF.EPORTRX.EPRX%dDATARATE" % i, widths[width])

    def sca_setup(self, verbose=False):
        # should this live here? I suppose so...
        for reg in self.DAQ_LPGBT.ec_config:
            if verbose:
                print(f"{reg}: {self.DAQ_LPGBT.ec_config[reg]}")
            self.DAQ_LPGBT.wr_reg(reg, self.DAQ_LPGBT.ec_config[reg])

    def sca_hard_reset(self):
        # should this live here? I suppose so...
        # NOTE this is potentially dangerous if GPIO 0 does something on RBv3
        # Luckily, it is only a MUX config pin we are toggling
        bit = 0
        self.DAQ_LPGBT.set_gpio(bit, 0)
        sleep(0.1)
        self.DAQ_LPGBT.set_gpio(bit, 1)
        sleep(0.1)

    def ready_led(self, to=1):
        # print('############################ReadoutBoard ready_led Debug############################')
        print(self.ver)
        if self.ver > 2 and self.trigger:
            self.TRIG_LPGBT.set_gpio("LED_1", to)
        else:
            self.DAQ_LPGBT.set_gpio("LED_1", to)

    def is_configured(self):
        # print('############################ReadoutBoard is_configured Debug############################')
        if self.configured:
            self.ready_led()
        
    def find_uplink_alignment(self, scan_time=0.01, default=0, data_mode=False, etroc='ETROC1'):  # default scan time of 0.01 is enough
        # TODO: check the FEC mode and set the number of links appropriately
        n_links = 24  #  NOTE: there are 28 e-links if the board is in FEC5 mode, but we are operating in FEC12 where there are only 24
        print ("Scanning for uplink alignment")
        print ("In data mode:", data_mode)
        alignment = {}
        inversion = {} # also scan for inversion
        # make alignment dict
        for link in ['Link 0', 'Link 1']:
            alignment[link] = {i:default for i in range(n_links)}
            inversion[link] = {i:0x02 for i in range(n_links)}

        # TODO: the scan should check the pattern checkers first, and skip the scan for any where the pattern check is already ok

        # now, scan
        if data_mode:
            for channel in range(n_links):
                res_daq = 0
                res_trig = 0
                for inv in [False, True]:
                    for shift in range(8):
                        self.DAQ_LPGBT.set_uplink_alignment(channel, shift, quiet=True)
                        self.DAQ_LPGBT.set_uplink_invert(channel, inv)
                        tmp = self.check_data_integrity(channel=channel, etroc=etroc, trigger=False)
                        if tmp>res_daq and tmp>1:  # NOTE: sometimes we find a random good word
                            print ("Found improved uplink alignment for Link 0, channel %s: %s, inverted: %s"%(channel, shift, inv))
                            alignment['Link 0'][channel] = shift
                            inversion['Link 0'][channel] = inv
                            res_daq = tmp
                        if self.trigger:
                            self.TRIG_LPGBT.set_uplink_alignment(channel, shift, quiet=True)
                            self.TRIG_LPGBT.set_uplink_invert(channel, inv)
                            tmp = self.check_data_integrity(channel=channel, etroc=etroc, trigger=True)
                            if tmp>res_trig and tmp>1:  # NOTE: sometimes we find a random good word
                                print ("Found improved uplink alignment for Link 1, channel %s: %s, inverted: %s"%(channel, shift, inv))
                                alignment['Link 1'][channel] = shift
                                inversion['Link 1'][channel] = inv
                                res_trig = tmp
        else:
            for inv in [False, True]:
                for shift in range(8):
                    for channel in range(n_links):
                        self.DAQ_LPGBT.set_uplink_alignment(channel, shift, quiet=True)
                        self.DAQ_LPGBT.set_uplink_invert(channel, inv)
                        if self.trigger:
                            self.TRIG_LPGBT.set_uplink_alignment(channel, shift, quiet=True)
                            self.TRIG_LPGBT.set_uplink_invert(channel, inv)
                    self.DAQ_LPGBT.set_uplink_group_data_source("normal")  # actually needed??
                    self.DAQ_LPGBT.set_downlink_data_src('upcnt')
                    self.DAQ_LPGBT.reset_pattern_checkers()
                    sleep(scan_time)
                    res = self.DAQ_LPGBT.read_pattern_checkers(log_dir=None, quiet=True)
                    for link in ['Link 0', 'Link 1']:
                        for channel in range(n_links):
                            if res[link]['UPCNT'][channel]['error'][0] == 0:
                                print ("Found uplink alignment for %s, channel %s: %s, inverted: %s"%(link, channel, shift, inv))
                                alignment[link][channel] = shift
                                inversion[link][channel] = inv

        # Reset alignment to default values for the channels where no good alignment has been found
        print ("Now setting uplink alignment to optimal values (default values if no good alignment was found)")
        for channel in range(n_links):
            self.DAQ_LPGBT.set_uplink_alignment(channel, alignment['Link 0'][channel], quiet=True)
            self.DAQ_LPGBT.set_uplink_invert(channel, inversion['Link 0'][channel])
            if self.trigger:
                self.TRIG_LPGBT.set_uplink_alignment(channel, alignment['Link 1'][channel], quiet=True)
                self.TRIG_LPGBT.set_uplink_invert(channel, inversion['Link 1'][channel])

        return alignment

    def dump_uplink_alignment(self, n_links=24):

        alignment = {
            'daq': {
                'alignment': {},
                'inversion': {},
            },
            'trigger': {
                'alignment': {},
                'inversion': {},
            }
        }

        for i in range(n_links):
            alignment['daq']['alignment'][i] = self.DAQ_LPGBT.get_uplink_alignment(i)
            alignment['daq']['inversion'][i] = self.DAQ_LPGBT.get_uplink_invert(i)

            if self.trigger:
                alignment['trigger']['alignment'][i] = self.TRIG_LPGBT.get_uplink_alignment(i)
                alignment['trigger']['inversion'][i] = self.TRIG_LPGBT.get_uplink_invert(i)

        return alignment

    def load_uplink_alignment(self, alignment, n_links=24):
        
        for i in range(n_links):
            self.DAQ_LPGBT.set_uplink_alignment(i ,alignment['daq']['alignment'][i])
            self.DAQ_LPGBT.set_uplink_invert(i, alignment['daq']['inversion'][i])

            if self.trigger:
                self.TRIG_LPGBT.set_uplink_alignment(i, alignment['trigger']['alignment'][i])
                self.TRIG_LPGBT.set_uplink_invert(i, alignment['trigger']['inversion'][i])

    def status(self):
        nodes = list(map (lambda x : "READOUT_BOARD_%s.LPGBT." % self.rb + x,
                          ("DOWNLINK.READY",
                      "UPLINK_0.READY",
                      "UPLINK_0.FEC_ERR_CNT",
                      "UPLINK_1.READY",
                      "UPLINK_1.FEC_ERR_CNT",)))
        for node in nodes:
            val = self.kcu.read_node(node)
            err = 0
            err |= ("READY" in node and val != 1)
            err |= ("FEC_ERR_CNT" in node and val != 0)
            if err:
                self.kcu.print_reg(self.kcu.hw.getNode(node), use_color=True, invert=True)

    def check_data_integrity(self, channel=0, etroc='ETROC1', trigger=False):
        '''
        Not sure where this function should live.
        It's not necessarily a part of the RB.
        '''
        from tamalero.FIFO import FIFO
        from tamalero.DataFrame import DataFrame
        df = DataFrame(etroc)
        lpgbt = 1 if trigger else 0
        fifo = FIFO(self, links=[{'elink': channel, 'lpgbt': lpgbt}], ETROC=etroc,)
        fifo.set_trigger(
            df.get_trigger_words(),
            df.get_trigger_masks(),
        )
        fifo.reset(l1a=True)
        n_header = 0
        n_trailer = 0
        data  = []
        for i in range(1):
            data += fifo.giant_dump(3000, align=False, format=False)  # + ['35', '55'] + fifo.giant_dump(3000)
            fifo.reset(l1a=True)

        good_counter = 0
        for word in data:
            word_type, _ = df.read(word, quiet=True)
            if word_type == None:
                return 0
            elif word_type in ['header', 'filler']:
                # ETROC2 data and filler definitions are so weak, it's easy to accidentially find them.
                good_counter += 1

        return good_counter


    def get_FEC_error_count(self, quiet=False):
        if not quiet:
            print("{:<8}{:<8}{:<50}{:<8}".format("Address", "Perm.", "Name", "Value"))
            self.kcu.print_reg(self.kcu.hw.getNode("READOUT_BOARD_%s.LPGBT.UPLINK_0.FEC_ERR_CNT" % self.rb), use_color=True, invert=True)
            self.kcu.print_reg(self.kcu.hw.getNode("READOUT_BOARD_%s.LPGBT.UPLINK_1.FEC_ERR_CNT" % self.rb), use_color=True, invert=True)
        return {
            'DAQ': self.kcu.read_node("READOUT_BOARD_%s.LPGBT.UPLINK_0.FEC_ERR_CNT" % self.rb).value(),
            'TRIGGER': self.kcu.read_node("READOUT_BOARD_%s.LPGBT.UPLINK_1.FEC_ERR_CNT" % self.rb).value()
        }

    def reset_FEC_error_count(self, quiet=False):
        if not quiet:
            print("Error counts before reset:")
            self.get_FEC_error_count()
        self.kcu.write_node("READOUT_BOARD_%s.LPGBT.FEC_ERR_RESET" % self.rb, 0x1)
        if not quiet:
            print("Error counts after reset:")
            self.get_FEC_error_count()

    def enable_rhett(self):
        self.DAQ_LPGBT.set_gpio('LED_RHETT', 1)

    def disable_rhett(self):
        self.DAQ_LPGBT.set_gpio('LED_RHETT', 0)

    def bad_boy(self, m=1):
        for x in range(60):
            self.enable_rhett()
            sleep(m*(0.000 + 0.0005*x))
            self.disable_rhett()
            sleep(m*0.005)
        self.enable_rhett()
        sleep(1)
        for x in range(60):
            self.enable_rhett()
            sleep(m*(0.030 - 0.0005*x))
            self.disable_rhett()
            sleep(m*0.005)
        self.disable_rhett()
        sleep(1)

    def configure(self):

        # configure the VTRX
        self.VTRX.configure(trigger=self.trigger)

        # the logic here is as follows:
        # dict -> load the provided alignment
        # none -> rerun alignment scan
        # anything else (e.g. False) -> don't touch the uplink alignment
        if isinstance(self.alignment, dict):
            self.load_uplink_alignment(self.alignment)
        elif self.alignment is None:
            _ = self.find_uplink_alignment(data_mode=self.data_mode, etroc=self.etroc)
        else:
            pass

        # SCA init (only if version is newer than 2)
        #self.
        # if self.ver < 3:
        #    self.sca_hard_reset()
        #    self.sca_setup(verbose=self.verbose)
        #    self.SCA.reset()
        #    self.SCA.connect()
        #    try:
        #        #print("version in SCA", self.SCA.ver)
        #        #print("config in SCA", self.SCA.config)
        #        self.SCA.config_gpios()  # this sets the directions etc according to the mapping
        #    except TimeoutError:
        #        print ("SCA config failed. Will continue without SCA.")

        #if self.trigger:
        #    self.DAQ_LPGBT.reset_trigger_mgts() 

        #sleep(0.5)

    def reset_link(self, trigger=False):
        '''
        Resets the links entirely, different procedure is necessary for production / prototype version of VTRX
        '''
        if trigger:
            if self.VTRX.ver == 'production':
                self.VTRX.reset(toggle_channels=[1])
            elif self.VTRX.ver == 'prototype':
                self.VTRX.reset(toggle_channels=[1])
            else:
                print ("Don't know how to reset VTRX version", self.VTRX.ver)
            self.VTRX.configure(trigger=trigger)
            self.DAQ_LPGBT.reset_trigger_mgts()
            self.TRIG_LPGBT.power_up_init()
        # else:
        #     if self.VTRX.ver == 'production':
        #         self.VTRX.reset()
        #     elif self.VTRX.ver == 'prototype':
        #         #self.VTRX.reset(toggle_channels=[0])
        #         self.VTRX.reset()
        #     else:
        #         print ("Don't know how to reset VTRX version", self.VTRX.ver)
        #     self.VTRX.configure(trigger=trigger)
        #     self.DAQ_LPGBT.reset_daq_mgts()
        #     self.DAQ_LPGBT.power_up_init()

        self.reset_FEC_error_count(quiet=True)

    def reset_problematic_links(self, max_retries=10, allow_bad_links=False):
        '''
        First check DAQ link, then trigger link.
        '''
        for link in ['DAQ', 'Trigger'] if self.trigger else ['DAQ']:
            for i in range(max_retries):
                sleep(0.01)  # this is actually needed for low error rates
                if link == 'DAQ':
                    good_link = self.DAQ_LPGBT.link_status()
                else:
                    good_link = self.TRIG_LPGBT.link_status()
                if good_link:
                    if self.verbose:
                        print (f"No FEC errors detected on {link} link")
                    break
                else:
                    if self.ver == 1:
                        pass
                    else:
                        self.reset_link(trigger = (link=='Trigger'))
                if i+2 > max_retries:
                    if allow_bad_links:
                        print (f"{link} link does not have a stable connection. Ignoring.")
                    else:
                        raise RuntimeError(f"{link} link does not have a stable connection after {max_retries} retries")
    

    ################################################################################
    # MUX64 chip configuration, not used in ETROC2_CE
    ################################################################################
    # # Uses MUX64 configuration to convert integer to voltage
    # def volt_conver_mux64(self,num,ch):
    #     voltage = (num / (2**12 - 1) ) * self.configuration['mux64']['input'][ch]['conv']
    #     return voltage
    
    # def read_mux_test_board(self, ch):
    #     #checks to make sure that the mux64 configuration is available
    #     assert 'mux64' in self.configuration, "MUX64 configuration not correctly loaded: Check configuration"
    #     #checks to see RB mapping version (RB ver = 1, SCA ver = RB.ver + 1)
    #     assert self.SCA.ver in [2], f"MUX64 testboard only works with 2v RB\nRB 1v detected"

    #     #channel select
    #     s0 = (ch & 0x01)
    #     s1 = (ch & 0x02) >> 1
    #     s2 = (ch & 0x04) >> 2
    #     s3 = (ch & 0x08) >> 3
    #     s4 = (ch & 0x10) >> 4
    #     s5 = (ch & 0x20) >> 5

    #     self.SCA.set_gpio('mux_addr0', s0) #mod_d08
    #     self.SCA.set_gpio('mux_addr1', s1) #mod_d09
    #     self.SCA.set_gpio('mux_addr2', s2) #mod_d10
    #     self.SCA.set_gpio('mux_addr3', s3) #mod_d11
    #     self.SCA.set_gpio('mux_addr4', s4) #mod_d12
    #     self.SCA.set_gpio('mux_addr5', s5) #mod_d13

    #     #read integar value and covert it to a voltge
    #     integer_volt = self.SCA.read_adc(0x12)
    #     gi = self.volt_conver_mux64(integer_volt,ch)

    #     return gi
    
    # def read_all_mux64_data(self, show = False):
    #     #checks to make sure that the mux64 configuration is available
    #     assert 'mux64' in self.configuration, "MUX64 configuration not correctly loaded: Check configuration"
    #     table=[]
    #     v_data=[]
    #     for i in self.configuration['mux64']['input'].keys():
    #         if self.configuration['mux64']['input'][i]['terminal_input']:
    #             volt=self.read_mux_test_board(i)
    #             v_data.append(volt)
    #         else:
    #             volt=None
    #             v_data.append(volt)
    #         sig_name=self.configuration['mux64']['input'][i]['sig_name']
    #         table.append([i, volt, sig_name])
        
    #     if (show):
    #         print(tabulate(table, headers=["Channel","Voltage", "Sig_Name"],  tablefmt="simple_outline"))

    #     return table

    # def read_selected_mux64(self, chs):
    #     volts = {}
    #     for i in chs:
    #         volts[str(i)] = self.read_mux_test_board(i)
    #     return volts

    # def mux64_monitoring(self, chs, tmax = 60, lat = 5, plot = False):
    #     # time is given in seconds
    #     mntr = {}
    #     mntr['record'] = []
    #     t = 0.0
    #     while (t<tmax):
    #         if (t%(60*5)==0):
    #             print(f'Monitoring: {t}/{tmax} secs')
    #         try:
    #             mntr['record'].append(self.read_selected_mux64(chs))
    #             mntr['record'][-1]['time'] = datetime.datetime.now().isoformat()
    #             time.sleep(lat)
    #         except:
    #             print("NonValidatedMemory exception: sleeping 0.1 secs...")
    #             time.sleep(0.1)
    #         t+=lat
    #     with open("mux64_mntr_%.2fmin.json".format(tmax/60.0), "w") as f:
    #         json.dump(mntr, f)
    #     if plot:
    #         fig, ax = plt.subplots(figsize=(10, 4))
    #         plt.title("MUX64 monitoring")
    #         plt.xlabel("Time")
    #         plt.ylabel("Voltage (V)")
    #         for ch in chs:
    #             vec = [mntr['record'][x][str(ch)] for x in range(len(mntr['record']))]
    #             vtime = [datetime.datetime.fromisoformat(mntr['record'][x]['time']) for x in range(len(mntr['record']))]
    #             plt.plot(vtime, vec, '.-', label=f"Channel: {ch}")
    #         locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    #         formatter = mdates.ConciseDateFormatter(locator)
    #         ax.xaxis.set_major_locator(locator)
    #         ax.xaxis.set_major_formatter(formatter)
    #         plt.grid(True)
    #         plt.legend(loc='best')
    #         fig.savefig('mux64_mntr_{:.2f}min.png'.format(tmax/60.0), dpi=600)
    #         plt.close(fig) 
    #     return 1

    def read_vtrx_temp(self):
        return self.VTRX.get_temp()

    def read_rb_thermistor(self, rt):

        # read voltage reference; some temperature readings depend on it
        v_ref = self.DAQ_LPGBT.read_dac()
        if v_ref==0 and ((rt==1 and self.ver==1) or rt==2):
            raise Exception("Read temperature called with VREF configured as 0V. VREF must be configured to read the temperatures.")

        if rt==1:

            if self.ver == 1:
                # This uses the DAC output for current so just read the voltage
                #current_rt1 = self.DAQ_LPGBT.set_current_dac_uA(0)  # make sure the current source is turned OFF in ver 1
                rt1_voltage = self.DAQ_LPGBT.read_adc(7)/(2**10-1) # FIXME: 7 should not be hardcoded
                return get_temp(rt1_voltage, v_ref, 10000, 25, 10000, 3900)  # this comes from the lpGBT ADC
            elif self.ver == 2:
                # Set the DAC current then read the voltage
                current_rt1 = self.DAQ_LPGBT.set_current_dac_uA(50)
                rt1_voltage = self.DAQ_LPGBT.read_adc(7)/(2**10-1) # FIXME: 7 should not be hardcoded
                return get_temp_direct(rt1_voltage, current_rt1, thermistor="NTCG063JF103FTB")  # this comes from the lpGBT ADC
            elif self.ver == 3:
                rt1_voltage = self.DAQ_LPGBT.read_adc(7)/(2**10-1) # FIXME: 7 should not be hardcoded
                #return get_temp(rt1_voltage, v_ref, 10000, 0, 0, 0, thermistor="NTCG063UH103HTBX")  # this comes from the lpGBT ADC
                return get_temp(rt1_voltage, v_ref, 10000, 25, 10000, 3900)  # this comes from the lpGBT ADC
            else:
                raise Exception("Unknown lpgbt version")

        elif rt==2:
            vref = 1.2
            rt2_voltage = self.DAQ_LPGBT.read_adc(1)/(2**10-1)
            return get_temp(rt2_voltage, vref, 10000, 25, 10000, 3900)  # this comes from the lpGBT ADC

            # if self.ver == 1:
                # rt2_voltage = self.SCA.read_adc(29)/(2**12-1) # FIXME: 29 should not be hardcoded
                # https://www.digikey.com/en/products/detail/tdk-corporation/NTCG063UH103HTBX/8565486
                # return get_temp(rt2_voltage, v_ref, 10000, 25, 10000, 3900)  # this comes from the SCA ADC
            # elif self.ver == 2:
                # rt2_voltage = self.SCA.read_adc(29)/(2**12-1) # FIXME: 29 should not be hardcoded
                # https://www.digikey.com/en/products/detail/tdk-corporation/NTCG063JF103FTB/5872743
                # return get_temp(rt2_voltage, v_ref, 10000, 25, 10000, 3380)  # this comes from the SCA ADC
            # elif self.ver == 3:
                # vref = 1.2
                # self.MUX64.select_channel(1)
                # rt2_voltage = self.DAQ_LPGBT.read_adc(1)/(2**10-1)
                # if False:
                    # this is the direct way of reading the voltage on RT2
                    # self.MUX64.select_channel(0)  # make sure you're not selecting channel 1!
                    # rt2_voltage = self.DAQ_LPGBT.read_adc(5)/(2**10-1)
                # return get_temp(rt2_voltage, vref, 10000, 25, 10000, 3900)  # this comes from the lpGBT ADC
            # else:
                # raise Exception("Unknown lpgbt version")

        else:
            raise Exception(f"Attempt to read unknown thermistor rt={rt}")

    def read_temp(self, verbose=False):

        """
        read all the temperature sensors
        """

        # internal temp from SCA
        t_vtrx = self.read_vtrx_temp()
        t_rt1 = self.read_rb_thermistor(1)
        t_rt2 = self.read_rb_thermistor(2)
        # if self.ver < 3:
        #     t_sca = self.SCA.read_temp()

        if verbose:
            print ("\nTemperature on RB RT1 is: %.1f C" % t_rt1)
            print ("Temperature on RB RT2 is: %.1f C" % t_rt2)
            # if self.ver < 3:
                # print ("Temperature on RB SCA is: %.1f C" % t_sca)
            # if self.ver > 1:
            #     print ("Temperature on RB VTRX is: %.1f C" % t_vtrx)

        res = {'t1': t_rt1, 't_VTRX': t_vtrx}
        if self.ver < 3:
            res['t2'] = t_rt2
            # res['t_SCA'] = t_sca
        return res

    def etroc_locked(self, elink, slave=False):
        if slave:
            locked = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_LOCKED_SLAVE").value()
        else:
            locked = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_LOCKED").value()

        return ((locked & (1 << elink)) >> elink) == True

    def disable_etroc_readout(self, elink=-1, slave=False, all=False):
        if all:
            self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE", 0x0FFFFFFF)
            self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE", 0x0FFFFFFF)
        elif elink>=0 and elink<28:
            if slave:
                disabled = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE").value()
                to_disable = disabled | (1 << elink)
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE", to_disable)
            else:
                disabled = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE").value()
                to_disable = disabled | (1 << elink)
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE", to_disable)
        else:
            print(f"Don't know what to do with elink number {elink}")

    def enable_etroc_readout(self, only=None, slave=False):
        if slave:
            if only is not None:
                disabled = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE").value()
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE", disabled ^ (1 << only))
            else:
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE_SLAVE", 0)
        else:
            if only is not None:
                disabled = self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE").value()
                #print(bin(disabled))
                #print(bin(disabled ^ (1 << only)))
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE", disabled ^ (1 << only))
            else:
                self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ETROC_DISABLE", 0)

    def reset_data_error_count(self):
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.PACKET_CNT_RESET", 0x1)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.ERR_CNT_RESET", 0x1)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.DATA_CNT_RESET", 0x1)

    def read_filler_rate(self, elink, slave=False):
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_ELINK_SEL0", elink)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_LPGBT_SEL0", 1 if slave else 0)
        return self.kcu.read_node(f"READOUT_BOARD_{self.rb}.FILLER_RATE").value()

    def read_packet_count(self, elink, slave=False):
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_ELINK_SEL0", elink)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_LPGBT_SEL0", 1 if slave else 0)
        return self.kcu.read_node(f"READOUT_BOARD_{self.rb}.PACKET_CNT").value()

    def read_error_count(self, elink, slave=False):
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_ELINK_SEL0", elink)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_LPGBT_SEL0", 1 if slave else 0)
        return self.kcu.read_node(f"READOUT_BOARD_{self.rb}.ERROR_CNT").value()

    def read_data_count(self, elink, slave=False):
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_ELINK_SEL0", elink)
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.FIFO_LPGBT_SEL0", 1 if slave else 0)
        return self.kcu.read_node(f"READOUT_BOARD_{self.rb}.DATA_CNT").value()

    def get_link_status(self, elink, slave=False, verbose=True):
        expected_filler_rate = 16500000

        locked = self.etroc_locked(elink, slave=slave)

        filler_rate = self.read_filler_rate(elink, slave=slave)
        error_count = self.read_error_count(elink, slave=slave)
        packet_count = self.read_packet_count(elink, slave=slave)
        data_count = self.read_data_count(elink, slave=slave)

        if verbose:
            print(f"- Status of link {elink}")
            colored = green if (locked) else red
            print(colored("{:20}{:10}".format("Locked", locked)))
            colored = green if (filler_rate > expected_filler_rate) else red
            print(colored("{:20}{:10}".format("Filler rate", filler_rate)))
            colored = green if (error_count < 1) else red
            print(colored("{:20}{:10}".format("Error count", error_count)))
            print("{:20}{:10}".format("Packet count", packet_count))
            print("{:20}{:10}".format("Data count", data_count))

            if filler_rate < expected_filler_rate:
                print("Filler rate is low. Try resetting PLL and FC of the ETROC.")

        return locked & (filler_rate > expected_filler_rate) & (error_count < 1)

    def enable_bitslip(self):
        self.kcu.write_node("READOUT_BOARD_%s.BITSLIP_AUTO_EN"%self.rb, 0x1)

    def disable_bitslip(self):
        self.kcu.write_node("READOUT_BOARD_%s.BITSLIP_AUTO_EN"%self.rb, 0x0)

    def rerun_bitslip(self):
        self.enable_bitslip()
        sleep(0.01)
        self.disable_bitslip()

    def enable_external_trigger(self):
        '''
        this lives in RB even though it's a system wide setting.
        Therefore, we let the user know.
        '''
        print("Enabling the external trigger for the KCU board")
        self.kcu.write_node("SYSTEM.EN_EXT_TRIGGER", 0x1)

    def disable_external_trigger(self):
        '''
        this lives in RB even though it's a system wide setting.
        Therefore, we let the user know.
        '''
        print("Disabling the external trigger for the KCU board")
        self.kcu.write_node("SYSTEM.EN_EXT_TRIGGER", 0x0)

    def external_trigger_status(self):
        status = "enabled" if self.kcu.read_node("SYSTEM.EN_EXT_TRIGGER") else "disabled"
        print(f"External trigger is currently {status}")

        return status == "enabled"

    def get_event_count(self):
        '''
        return number of L1As that went through this board
        '''
        return self.kcu.read_node(f"READOUT_BOARD_{self.rb}.EVENT_CNT").value()

    def reset_event_count(self):
        '''
        reset the event counter
        '''
        self.kcu.write_node(f"READOUT_BOARD_{self.rb}.EVENT_CNT_RESET", 0x1)

    def connect_modules(self, power_board=False, moduleids=[9996,9997,9998,9999], hard_reset=False, ext_vref=False, verbose=False):
        self.modules = []
        for i in range(self.nmodules):
            if verbose: print(f"Working on module {i}")
            if i >= len(moduleids): break
            self.modules.append(
                Module(
                    self,
                    i+1,
                    enable_power_board=power_board,
                    moduleid=moduleids[i],
                    hard_reset = hard_reset,
                    ext_vref=ext_vref,
                    verbose=verbose,
                ),
            )
            if self.modules[-1].connected:
                print(f"Readout Board {self.rb}: Found connected Module {i+1}")

    def dark_mode(self):
        self.DAQ_LPGBT.set_gpio("LED_RHETT", 0)  # rhett
        self.DAQ_LPGBT.set_gpio("LED_0", 0)  # Set LED0 after succesfull gpio configure
        self.DAQ_LPGBT.set_gpio("LED_1", 0) # Set LED1 after tamalero finishes succesfully
        # if self.ver < 3:
        #     self.SCA.set_gpio("sca_led", 0)

    def light_mode(self):
        self.DAQ_LPGBT.set_gpio("LED_RHETT", 1)  # rhett
        self.DAQ_LPGBT.set_gpio("LED_0", 1)  # Set LED0 after succesfull gpio configure
        self.DAQ_LPGBT.set_gpio("LED_1", 1) # Set LED1 after tamalero finishes succesfully
        # if self.ver < 3:
        #     self.SCA.set_gpio("sca_led", 1)
