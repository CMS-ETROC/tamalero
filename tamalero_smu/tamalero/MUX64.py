import os
import random
import time, datetime
import json
from tamalero.utils import read_mapping, get_config
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    from tabulate import tabulate
    has_tabulate = True
except ModuleNotFoundError:
    print ("Package `tabulate` not found.")
    has_tabulate = False

def channel_byname(channel_func):
    @wraps(channel_func)
    def wrapper(mux64, channel, calibrate):
        if isinstance(channel, str):
            channel_dict = mux64.channel_mapping
            pin = channel_dict[channel]['pin']
            return channel_func(mux64, pin, calibrate)
        elif isinstance(channel, int):
            return channel_func(mux64, channel, calibrate)
        else:
            invalid_type = type(channel)
            raise TypeError(f"{channel_func.__name__} can only take positional arguments of type int or str, but argument of type {invalid_type} was given.")
    return wrapper

def channel_bypin(channel_func):
    @wraps(channel_func)
    def wrapper(mux64, value, channel, direct):
        if isinstance(channel, int):
            channel_dict = mux64.channel_mapping
            for ch in channel_dict.keys():
                if channel_dict[ch]["pin"] == channel:
                    name = ch
                    break
            return channel_func(mux64, value, ch, direct)
        elif isinstance(channel, str):
            return channel_func(mux64, value, channel, direct)
        else:
            invalid_type = type(channel)
            raise TypeError(f"{channel_func.__name__} can only take positional arguments of type int or str, but argument of type {invalid_type} was given.")
    return wrapper

class MUX64:

    def __init__(self, rb=0, ver=0, config='default', rbver=None, LPGBT=None, SCA=None):
        self.rb = rb
        self.ver = 1  # NOTE same as for SCA, we're giving it the lpGBT version (bur for the moment not used)
        self.config = config
        self.LPGBT = LPGBT
        self.SCA = SCA
        self.rbver = rbver

        self.configure()

        if self.rbver and self.rbver < 3:
            print("This MUX64 is associated to an old version of the Readout Board")

        if LPGBT and SCA:
            print("MUX64 is connected to both LPGBT and SCA: Please pick one")


    def is_connected(self):
        if self.LPGBT:
            print("Connected to LPGBT")
            self.LPGBT.init_adc()
            return 1
        elif self.SCA:
            print("Connected to SCA")
            return 1
        else:
            return 0


    def configure(self):
        self.set_channel_mapping()
        if self.LPGBT:
            for p in range(1, 6+1):
                self.LPGBT.set_gpio_direction(f"MUXCNT{p}", 1)


    def set_channel_mapping(self):
        self.channel_mapping = get_config(self.config, version=f'v{self.rbver}')['MUX64']['channel']
   
    @channel_bypin
    def volt_conver(self,value,channel,direct):
        if self.SCA:
            voltage = (value / (2**12 - 1) )
            if not direct:
                voltage = voltage * self.channel_mapping[channel]['conv']
        elif self.LPGBT:
            #value_calibrated = value * self.LPGBT.cal_gain / 1.85 + (512 - self.LPGBT.cal_offset)
            #input_voltage = value_calibrated / (2**10 - 1) * self.LPGBT.adc_mapping['MUX64OUT']['conv']
            voltage = value/(2**10 - 1)
            if not direct:
                voltage = voltage * self.get_conversion_factor(R1=self.channel_mapping[channel]['R1'], R2=self.channel_mapping[channel]['R2'])
        else:
            voltage = 0.0
        return voltage

    def select_channel(self, channel):
        s0 = (channel & 0x01)
        s1 = (channel & 0x02) >> 1
        s2 = (channel & 0x04) >> 2
        s3 = (channel & 0x08) >> 3
        s4 = (channel & 0x10) >> 4
        s5 = (channel & 0x20) >> 5

        if self.SCA:
            self.SCA.set_gpio('mux_addr0', s0)
            self.SCA.set_gpio('mux_addr1', s1)
            self.SCA.set_gpio('mux_addr2', s2)
            self.SCA.set_gpio('mux_addr3', s3)
            self.SCA.set_gpio('mux_addr4', s4)
            self.SCA.set_gpio('mux_addr5', s5)

        if self.LPGBT:
            self.LPGBT.set_gpio('MUXCNT1', s0)
            self.LPGBT.set_gpio('MUXCNT2', s1)
            self.LPGBT.set_gpio('MUXCNT3', s2)
            self.LPGBT.set_gpio('MUXCNT4', s3)
            self.LPGBT.set_gpio('MUXCNT5', s4)
            self.LPGBT.set_gpio('MUXCNT6', s5)
    
    @channel_byname
    def read_adc(self, channel, calibrate=False):

        #channel select
        self.select_channel(channel)

        if self.SCA:
            value = self.SCA.read_adc(0x12)

        if self.LPGBT:
            value = self.LPGBT.read_adc(self.LPGBT.adc_mapping['MUX64OUT']['pin'], calibrate=calibrate)
        
        return value

    def read_channel(self, channel, calibrate=True, direct=False):
        value = self.read_adc(channel, calibrate=calibrate)
        value = self.volt_conver(value,channel, direct=direct)
        return value

    def read_channels(self): #read and print all adc values
        self.set_channel_mapping()
        channel_dict = self.channel_mapping
        table = []
        will_fail = False
        for channel in channel_dict.keys():
            pin = channel_dict[channel]['pin']
            comment = channel_dict[channel]['comment']
            value = self.read_adc(pin, calibrate=True)
            value_raw = self.read_adc(pin, calibrate=False)
            voltage = self.read_channel(pin)
            voltage_direct = self.read_channel(pin, direct=True)
            table.append([channel, pin, value_raw, value, voltage_direct, voltage, comment])

        headers = ["Channel","Pin", "Reading (raw)", "Reading (calib)", "Voltage (direct)", "Voltage (conv)", "Comment"]

        if has_tabulate:
            print(tabulate(table, headers=headers,  tablefmt="simple_outline"))
        else:
            header_string = "{:<20}"*len(headers)
            data_string = "{:<20}{:<20}{:<20.0f}{:<20.0f}{:<20.3f}{:<20.3f}{:<20}"
            print(header_string.format(*headers))
            for line in table:
                print(data_string.format(*line))

    def get_conversion_factor(self, R=0, R1=82, R2=82):
        '''
        resistance values in kOhm
        R - voltage divider on the MUX output
        R1 - resistor between measurement source and MUX input
        R2 - resistor between MUX input and ground
        '''
        if R==0:
            return (R1+R2)/R2
        else:
            return 1/((1/2)*(2*R*R2)/(R1*(2*R+R1)+2*R*R2))

    def monitor_channels(self, channels = ['mod0_a5', 'mod1_a5', 'mod2_a5'], lat = 1.0, time_max = 60.0, plot = True):
        # time is given in seconds
        self.set_channel_mapping()
        channel_dict = self.channel_mapping
        mntr = {}
        print(channels)
        for channel in channels:
            mntr[channel] = []
        mntr['time'] = []
        start_time = time.time()
        now_time = time.time()
        while ((now_time - start_time) < time_max):
            #if (t%(60*5)==0):
            #    print(f'Monitoring: {t}/{tmax} secs')
            try:
                for channel in channels:
                    pin = channel_dict[channel]['pin']
                    value = self.read_adc(pin, calibrate=True)
                    value_raw = self.read_adc(pin, calibrate=False)
                    voltage = self.read_channel(pin)
                    voltage_direct = self.read_channel(pin, direct=True)
                    mntr[channel].append(voltage)
                mntr['time'].append(datetime.datetime.now().isoformat())
                time.sleep(lat)
            except:
                print("NonValidatedMemory exception: sleeping 0.1 secs...")
                time.sleep(0.1)
            now_time = time.time()
        with open("mux64_mntr_%.2fmin.json".format(time_max/60.0), "w") as f:
            json.dump(mntr, f)
        if plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.title("MUX64 channel monitoring")
            plt.xlabel("Time")
            plt.ylabel("Voltage (V)")
            ymin = 9999.0
            ymax = 0.0
            for channel in channels:
                vec = [mntr[channel][x] for x in range(len(mntr[channel]))]
                vtime = [datetime.datetime.fromisoformat(mntr['time'][x]) for x in range(len(mntr['time']))]
                plt.plot(vtime, vec, '.-', label=f"Channel: {channel}")
                if min(vec) < ymin:
                    ymin = min(vec)
                if max(vec) > ymax:
                    ymax = max(vec)
            ax.set_ylim(0.98*ymin, 1.02*ymax)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            #print(ymin, ymax)
            plt.grid(True)
            plt.legend(loc='best')
            fig.savefig('rb3_mux64_mntr_{:.2f}min.png'.format(time_max/60.0), dpi=600)
            plt.close(fig) 
        return 1
