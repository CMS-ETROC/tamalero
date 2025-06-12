import os
import random
from tamalero.utils import read_mapping, get_config
from functools import wraps
import time
try:
    from tabulate import tabulate
    has_tabulate = True
except ModuleNotFoundError:
    print ("Package `tabulate` not found.")
    has_tabulate = False

class SCA_CRB:
    # 0 is reserved
    ENSPI  = 1
    ENGPIO = 2
    ENI2C0 = 3
    ENI2C1 = 4
    ENI2C2 = 5
    ENI2C3 = 6
    ENI2C4 = 7

class SCA_CRC:
    ENI2C5 = 0
    ENI2C6 = 1
    ENI2C7 = 2
    ENI2C8 = 3
    ENI2C9 = 4
    ENI2CA = 5
    ENI2CB = 6
    ENI2CC = 7

class SCA_CRD:
    ENI2CD = 0
    ENI2CE = 1
    ENI2CF = 2
    ENJTAG = 3
    ENADC  = 4
    ENDAC  = 6

class SCA_CONTROL:
    CTRL_R_ID  = 0x14D1  # this is SCA V2
    CTRL_W_CRB = 0x0002
    CTRL_R_CRB = 0x0003
    CTRL_W_CRC = 0x0004
    CTRL_R_CRC = 0x0005
    CTRL_W_CRD = 0x0006
    CTRL_R_CRD = 0x0007
    CTRL_R_SEU = 0x13F1
    CTRL_C_SEU = 0x13F1

class SCA_GPIO:
    GPIO_W_DATAOUT   = 0x0210
    GPIO_R_DATAOUT   = 0x0211
    GPIO_R_DATAIN    = 0x0201
    GPIO_W_DIRECTION = 0x0220
    GPIO_R_DIRECTION = 0x0221

class SCA_ADC:
    ADC_GO     = 0x1402
    ADC_W_MUX  = 0x1450
    ADC_R_MUX  = 0x1451
    ADC_W_CURR = 0x1460
    ADC_R_CURR = 0x1461
    ADC_W_GAIN = 0x1410
    ADC_R_GAIN = 0x1411
    ADC_R_DATA = 0x1421
    ADC_R_RAW  = 0x1431
    ADC_R_OFS  = 0x1441

class SCA_JTAG:
    # JTAG COMMANDS
    JTAG_W_CTRL = 0x1380
    JTAG_R_CTRL = 0x1381
    JTAG_W_FREQ = 0x1390
    JTAG_R_FREQ = 0x1391
    JTAG_W_TDO0 = 0x1300
    JTAG_R_TDI0 = 0x1301
    JTAG_W_TDO1 = 0x1310
    JTAG_R_TDI1 = 0x1311
    JTAG_W_TDO2 = 0x1320
    JTAG_R_TDI2 = 0x1321
    JTAG_W_TDO3 = 0x1330
    JTAG_R_TDI3 = 0x1331
    JTAG_W_TMS0 = 0x1340
    JTAG_R_TMS0 = 0x1341
    JTAG_W_TMS1 = 0x1350
    JTAG_R_TMS1 = 0x1351
    JTAG_W_TMS2 = 0x1360
    JTAG_R_TMS2 = 0x1361
    JTAG_W_TMS3 = 0x1370
    JTAG_R_TMS3 = 0x1371
    JTAG_ARESET = 0x13C0
    JTAG_GO     = 0x13A2
    JTAG_GO_M   = 0x13B0

class SCA_I2C:
    # I2C commands
    I2C_W_CTRL = 0x30 # write control register
    I2C_R_CTRL = 0x31 # read control register
    I2C_R_STR  = 0x11 # read status register
    I2C_S_7B_W = 0x82 # single byte write
    I2C_S_7B_R = 0x86 # single byte read
    I2C_M_7B_R = 0xDE # multi-byte read
    I2C_M_7B_W = 0xDA # multi-byte write
    I2C_W_DATA0 = 0x40 # write to data register 0
    I2C_W_DATA1 = 0x50 # write to data register 1
    I2C_W_DATA2 = 0x60 # write to data register 2
    I2C_W_DATA3 = 0x70 # write to data register 3
    I2C_R_DATA0 = 0x41 # read from data register 0
    I2C_R_DATA1 = 0x51 # read from data register 1
    I2C_R_DATA2 = 0x61 # read from data register 2
    I2C_R_DATA3 = 0x71 # read from data register 3
    I2C_RW_DATA_OFFSET = 16 # offset to access data register 1, 2, 3

def gpio_byname(gpio_func):
    @wraps(gpio_func)
    def wrapper(sca, pin, direction=1):
        if isinstance(pin, str):
            gpio_dict = sca.gpio_mapping
            pin = gpio_dict[pin]['pin']
            return gpio_func(sca, pin, direction)
        elif isinstance(pin, int):
            return gpio_func(sca, pin, direction)
        else:
            invalid_type = type(pin)
            raise TypeError(f"{gpio_func.__name__} can only take positional arguments of type int or str, but argument of type {invalid_type} was given.")

    return wrapper

class SCA:

    def __init__(self, rb=0, flavor='small', ver=0, config='default', poke=False, verbose=False):
        self.rb = rb
        self.flavor = flavor
        self.err_count = 0
        self.ver = ver + 1  # NOTE don't particularly like this, but we're giving it the lpGBT version
        self.config = config
        self.locked = False
        self.set_adc_mapping()
        self.set_gpio_mapping()
        self.verbose = verbose
        self.i2c_enabled = 0
        if poke:
            self.verbose = False

    def connect_KCU(self, kcu):
        self.kcu = kcu

    def set_adc_mapping(self):
        assert self.ver in [1, 2], f"Unrecognized version {self.ver}"
        self.adc_mapping = get_config(self.config, version=f'v{self.ver}')['SCA']['adc']

    def set_gpio_mapping(self):
        assert self.ver in [1, 2], f"Unrecognized version {self.ver}"
        self.gpio_mapping = get_config(self.config, version=f'v{self.ver}')['SCA']['gpio']

    def update_ver(self, new_ver):
        assert new_ver in [1, 2], f"Unrecognized version {new_ver}"
        self.ver = new_ver
        self.set_adc_mapping()
        self.set_gpio_mapping()

    def reset(self):
        self.kcu.action("READOUT_BOARD_%d.SC.START_RESET" % self.rb)
    
    def connect(self):
        self.kcu.action("READOUT_BOARD_%d.SC.START_CONNECT" % self.rb)

    def enable(self, state=1):
        self.kcu.write_node("READOUT_BOARD_%d.SC.SCA_ENABLE" % self.rb, state)

    def rw_reg(self, reg, data=0x0, adr=0x00, transid=0x00):
        cmd = reg & 0xFF
        channel = (reg >> 8) & 0xFF
        return self.rw_cmd(cmd, channel, data, adr, transid)

    def rw_cmd(self, cmd, channel, data, adr=0x0, transid=0x00, time_out=1.3, verbose=False):
        """
        adr = chip address (0x0 by default)
        """
        if transid == 0:
            transid = random.randint(1, 2**8-2)  # transid of 0 or 255 gives error

        # request packet structure
        # sof
        # address : destination packet address (chip id)
        # control : connect/sabm, reset, test
        # {
        #  transid
        #  channel
        #  length
        #  command
        #  data[31:0]
        # }
        # fcs
        # eof

        if verbose:
            print("SCA r/w:")
            print(f"transid={transid}, channel={channel}, cmd={cmd}, adr={adr}, data={data}")


        self.kcu.toggle_dispatch()

        self.kcu.write_node("READOUT_BOARD_%d.SC.TX_CHANNEL" % self.rb, channel)
        self.kcu.write_node("READOUT_BOARD_%d.SC.TX_CMD" % self.rb, cmd)
        self.kcu.write_node("READOUT_BOARD_%d.SC.TX_ADDRESS" % self.rb, adr)
    
        self.kcu.write_node("READOUT_BOARD_%d.SC.TX_TRANSID" % self.rb, transid)
    
        self.kcu.write_node("READOUT_BOARD_%d.SC.TX_DATA" % self.rb, data)
        self.kcu.action("READOUT_BOARD_%d.SC.START_COMMAND" % self.rb)

        self.kcu.dispatch()
    
        # reply packet structure
        # sof
        # address
        # control
        # {
        #  transid
        #  channel
        #  error
        #  length
        #  data
        # }
        # fcs
        # eof
    
        # TODO: read reply
        err = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_ERR" % self.rb).value()  # 8 bit
        if err > 0:
            if (err & 0x1):
                print("SCA Read Error :: Generic Error Flag")
            if (err & 0x2):
                print("SCA Read Error :: Invalid Channel Request")
            if (err & 0x4):
                print("SCA Read Error :: Invalid Command Request")
            if (err & 0x8):
                print("SCA Read Error :: Invalid Transaction Number Request")
            if (err & 0x10):
                print("SCA Read Error :: Invalid Length")
            if (err & 0x20):
                print("SCA Read Error :: Channel Not Enabled")
            if (err & 0x40):
                print("SCA Read Error :: Command In Treatment")

        self.kcu.toggle_dispatch()
        rx_rec  = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_RECEIVED" % self.rb)  # flag pulse
        rx_ch   = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_CHANNEL" % self.rb)  # channel reply
        rx_len  = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_LEN" % self.rb)
        rx_ad   = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_ADDRESS" % self.rb)
        rx_ctrl = self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_CONTROL" % self.rb)
        self.kcu.dispatch()

        # dispatch and get the read values
        rx_rec  = rx_rec.value()  # flag pulse
        rx_ch   = rx_ch.value()  # channel reply
        rx_len  = rx_len.value()
        rx_ad   = rx_ad.value()
        rx_ctrl = rx_ctrl.value()

        if verbose:
            print(f"Received: err={err}, rx_rec={rx_rec}, rx_ch={rx_ch}, rx_len={rx_len}, rx_ad={rx_ad}, rx_ctrl={rx_ctrl}")

        # NOTE I2C transaction can be slow, so try reading the transid several times before it times out
        start_time = time.time()
        while not transid == self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_TRANSID" % self.rb):
            self.kcu.write_node("READOUT_BOARD_%d.SC.RX_RESET" % self.rb, 0x01)
            self.kcu.write_node("READOUT_BOARD_%d.SC.TX_RESET" % self.rb, 0x01)
            if time.time() - start_time > time_out:
                if verbose:
                    print(f"data: {self.kcu.read_node('READOUT_BOARD_%d.SC.RX.RX_DATA' % self.rb)}")
                    print("SCA Read Error :: Transaction ID Does Not Match")
                    print("SCA Read Error :: Resetting RX/TX")
                raise TimeoutError("SCA Error :: Transaction timed out.")

        return self.kcu.read_node("READOUT_BOARD_%d.SC.RX.RX_DATA" % self.rb)  # 32 bit read data

    def read_control_registers(self, verbose=False):
        # don't need to read CRC
        crb_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRB) >> 24
        crc_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRC) >> 24
        crd_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRD) >> 24

        en_gpio = (crb_rd >> SCA_CRB.ENGPIO) & 1
        en_spi = (crb_rd >> SCA_CRB.ENSPI) & 1
        #en_i2c = (crb_rd >> )
        if verbose:
            print(f"SCA control registers: en_gpio={en_gpio}")
            print(f"SCA control registers: en_spi={en_spi}")
            print (crb_rd, crc_rd, crd_rd)
        return crb_rd, crc_rd, crd_rd

    def get_I2C_status(self, channel):
        channel_str = hex(channel).upper()[-1]
        if channel < 5:
            crb_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRB) >> 24
            return (crb_rd >> getattr(SCA_CRB, f"ENI2C{channel_str}")) & 0x1
        elif channel < 13:
            crc_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRC) >> 24
            return (crc_rd >> getattr(SCA_CRC, f"ENI2C{channel_str}")) & 0x1
        elif channel < 16:
            crd_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRD) >> 24
            return (crd_rd >> getattr(SCA_CRD, f"ENI2C{channel_str}")) & 0x1
        else:
            raise RuntimeError(f"SCA only has 16 I2C channels, don't know what to do with channel {channel}")

    def enable_I2C(self, channel=0):
        '''
        just enable a single i2c channel
        '''
        crb, crc, crd = self.read_control_registers()
        ENI2C0  = (channel == 0) & 0x1
        ENI2C1  = (channel == 1) & 0x1
        ENI2C2  = (channel == 2) & 0x1
        ENI2C3  = (channel == 3) & 0x1
        ENI2C4  = (channel == 4) & 0x1
        ENI2C5  = (channel == 5) & 0x1
        ENI2C6  = (channel == 6) & 0x1
        ENI2C7  = (channel == 7) & 0x1
        ENI2C8  = (channel == 8) & 0x1
        ENI2C9  = (channel == 9) & 0x1
        ENI2CA  = (channel == 10) & 0x1
        ENI2CB  = (channel == 11) & 0x1
        ENI2CC  = (channel == 12) & 0x1
        ENI2CD  = (channel == 13) & 0x1
        ENI2CE  = (channel == 14) & 0x1
        ENI2CF  = (channel == 15) & 0x1

        crb |= ENI2C0 << SCA_CRB.ENI2C0
        crb |= ENI2C1 << SCA_CRB.ENI2C1
        crb |= ENI2C2 << SCA_CRB.ENI2C2
        crb |= ENI2C3 << SCA_CRB.ENI2C3
        crb |= ENI2C4 << SCA_CRB.ENI2C4

        crc |= ENI2C5 << SCA_CRC.ENI2C5
        crc |= ENI2C6 << SCA_CRC.ENI2C6
        crc |= ENI2C7 << SCA_CRC.ENI2C7
        crc |= ENI2C8 << SCA_CRC.ENI2C8
        crc |= ENI2C9 << SCA_CRC.ENI2C9
        crc |= ENI2CA << SCA_CRC.ENI2CA
        crc |= ENI2CB << SCA_CRC.ENI2CB
        crc |= ENI2CC << SCA_CRC.ENI2CC

        crd |= ENI2CD << SCA_CRD.ENI2CD
        crd |= ENI2CE << SCA_CRD.ENI2CE
        crd |= ENI2CF << SCA_CRD.ENI2CF

        self.rw_reg(SCA_CONTROL.CTRL_W_CRB, crb << 24)
        self.rw_reg(SCA_CONTROL.CTRL_W_CRC, crc << 24)
        self.rw_reg(SCA_CONTROL.CTRL_W_CRD, crd << 24)

        self.i2c_enabled = self.i2c_enabled | ( 1 << channel )

    def enable_gpio(self):
        crb, crc, crd = self.read_control_registers()
        crb |= 1 << SCA_CRB.ENGPIO
        self.rw_reg(SCA_CONTROL.CTRL_W_CRB, crb << 24)

    def enable_spi(self):
        crb, crc, crd = self.read_control_registers()
        crb |= 1 << SCA_CRB.ENSPI
        self.rw_reg(SCA_CONTROL.CTRL_W_CRB, crb << 24)

    def enable_adc(self):
        crb, crc, crd = self.read_control_registers()
        crd |= 1 << SCA_CRD.ENADC
        self.rw_reg(SCA_CONTROL.CTRL_W_CRD, crd << 24)

    def enable_dac(self):
        crb, crc, crd = self.read_control_registers()
        crd |= 1 << SCA_CRD.ENDAC
        self.rw_reg(SCA_CONTROL.CTRL_W_CRD, crd << 24)

    def configure_control_registers(self, en_spi=0, en_gpio=0, en_i2c=0, en_adc=0, en_dac=0):
    
        ENI2C0  = (en_i2c >> 0) & 0x1
        ENI2C1  = (en_i2c >> 1) & 0x1
        ENI2C2  = (en_i2c >> 2) & 0x1
        ENI2C3  = (en_i2c >> 3) & 0x1
        ENI2C4  = (en_i2c >> 4) & 0x1
        ENI2C5  = (en_i2c >> 5) & 0x1
        ENI2C6  = (en_i2c >> 6) & 0x1
        ENI2C7  = (en_i2c >> 7) & 0x1
        ENI2C8  = (en_i2c >> 8) & 0x1
        ENI2C9  = (en_i2c >> 9) & 0x1
        ENI2CA  = (en_i2c >> 10) & 0x1
        ENI2CB  = (en_i2c >> 11) & 0x1
        ENI2CC  = (en_i2c >> 12) & 0x1
        ENI2CD  = (en_i2c >> 13) & 0x1
        ENI2CE  = (en_i2c >> 14) & 0x1
        ENI2CF  = (en_i2c >> 15) & 0x1
    
        crb = 0
        crb |= en_spi << SCA_CRB.ENSPI
        crb |= en_gpio << SCA_CRB.ENGPIO
        crb |= ENI2C0 << SCA_CRB.ENI2C0
        crb |= ENI2C1 << SCA_CRB.ENI2C1
        crb |= ENI2C2 << SCA_CRB.ENI2C2
        crb |= ENI2C3 << SCA_CRB.ENI2C3
        crb |= ENI2C4 << SCA_CRB.ENI2C4
    
        crc = 0
        crc |= ENI2C5 << SCA_CRC.ENI2C5
        crc |= ENI2C6 << SCA_CRC.ENI2C6
        crc |= ENI2C7 << SCA_CRC.ENI2C7
        crc |= ENI2C8 << SCA_CRC.ENI2C8
        crc |= ENI2C9 << SCA_CRC.ENI2C9
        crc |= ENI2CA << SCA_CRC.ENI2CA
        crc |= ENI2CB << SCA_CRC.ENI2CB
        crc |= ENI2CC << SCA_CRC.ENI2CC
    
        crd = 0
        crd |= ENI2CD << SCA_CRD.ENI2CD
        crd |= ENI2CE << SCA_CRD.ENI2CE
        crd |= ENI2CF << SCA_CRD.ENI2CF
        crd |= en_adc << SCA_CRD.ENADC
        crd |= en_dac << SCA_CRD.ENDAC

        self.rw_reg(SCA_CONTROL.CTRL_W_CRB, crb << 24)
        self.rw_reg(SCA_CONTROL.CTRL_W_CRC, crc << 24)
        self.rw_reg(SCA_CONTROL.CTRL_W_CRD, crd << 24)
    
        crb_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRB) >> 24
        crc_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRC) >> 24
        crd_rd = self.rw_reg(SCA_CONTROL.CTRL_R_CRD) >> 24
    
        if (crb != crb_rd or crc != crc_rd or crd != crd_rd):
            print("SCA Control Register Readback Error, Not configured Correctly")
            print("CRB wr=%02X, rd=%02X" % (crb, crb_rd))
            print("CRC wr=%02X, rd=%02X" % (crc, crc_rd))
            print("CRD wr=%02X, rd=%02X" % (crd, crd_rd))

    def read_adc_curr(self, pin):
        # just do one at a time, does not need to run constantly
        self.enable_adc() #enable ADC
        tmp = 1 << pin
        self.rw_reg(SCA_ADC.ADC_W_CURR, tmp)
        self.rw_reg(SCA_ADC.ADC_W_MUX, pin) #configure register we want to read
        #val = self.rw_reg(SCA_ADC.ADC_R_CURR).value()
        val = self.rw_reg(SCA_ADC.ADC_GO, 0x01).value() #execute and read ADC_GO command
        self.rw_reg(SCA_ADC.ADC_W_MUX, 0x0) #reset register to default (0)
        self.rw_reg(SCA_ADC.ADC_W_CURR, 0x0) #reset register to default (0)
        return val

    def disable_adc_curr(self):
        # disable current source for ALL channels
        self.enable_adc() #enable ADC
        self.rw_reg(SCA_ADC.ADC_W_CURR, 0)
        val = self.rw_reg(SCA_ADC.ADC_R_CURR).value()
        return val == 0

    def read_adc(self, pin = 0, raw=False):
        # either read raw ADC values for a pin, or physical quantity for
        # a (string) named port
        conv = 1
        if isinstance(pin, str):
            if not raw:
                conv = self.adc_mapping[pin]['conv'] / (2**12 - 1)
            pin = self.adc_mapping[pin]['pin']
        self.enable_adc() #enable ADC
        self.rw_reg(SCA_ADC.ADC_W_MUX, pin) #configure register we want to read
        val = self.rw_reg(SCA_ADC.ADC_GO, 0x01).value() #execute and read ADC_GO command
        self.rw_reg(SCA_ADC.ADC_W_MUX, 0x0) #reset register to default (0)
        return val*conv

    def read_adcs(self, check=False, strict_limits=False): #read and print all adc values
        adc_dict = self.adc_mapping
        table=[]
        will_fail = False
        for adc_reg in adc_dict.keys():
            pin = adc_dict[adc_reg]['pin']
            comment = adc_dict[adc_reg]['comment']
            value = self.read_adc(pin)
            input_voltage = value / (2**12 - 1) * adc_dict[adc_reg]['conv']
            if check:
                try:
                    min_v = adc_dict[adc_reg]['min']
                    max_v = adc_dict[adc_reg]['max']
                    status = "OK" if (input_voltage >= min_v) and (input_voltage <= max_v) else "ERR"
                    if status == "ERR" and strict_limits:
                        will_fail = True
                except KeyError:
                    status = "N/A"
                table.append([adc_reg, pin, value, input_voltage, status, comment])
            else:
                table.append([adc_reg, pin, value, input_voltage, comment])

        if check:
            headers = ["Register","Pin", "Reading", "Voltage", "Status", "Comment"]
        else:
            headers = ["Register","Pin", "Reading", "Voltage", "Comment"]

        if has_tabulate:
            print(tabulate(table, headers=headers,  tablefmt="simple_outline"))
        else:
            header_string = "{:<20}"*len(headers)
            data_string = "{:<20}{:<20}{:<20.0f}{:<20.3f}{:<20}"
            if check:
                data_string += "{:<20}"
            print(header_string.format(*headers))
            for line in table:
                print(data_string.format(*line))

        if will_fail:
            raise ValueError("At least one input voltage is out of bounds, with status ERR as seen in the table above")

    def read_temp(self):
        # not very precise (according to manual), but still useful.
        return ((self.read_adc(31)/2**12)*1000 - 716)/-1.829

    @gpio_byname
    def read_gpio(self, line, to=1):
        self.enable_gpio()  # enable GPIO
        val = self.rw_reg(SCA_GPIO.GPIO_R_DATAIN).value()
        return int((val >> line) & 1)

    @gpio_byname
    def set_gpio(self, line, to=1):
        self.enable_gpio()  # enable GPIO
        currently_set = self.rw_reg(SCA_GPIO.GPIO_R_DATAOUT).value()
        if (currently_set & (1 << line)) and to==0:
            currently_set ^= (1 << line)
        elif to==1:
            currently_set |= (1 << line)
        #self.rw_reg(SCA_GPIO.GPIO_W_DIRECTION, currently_set)
        self.rw_reg(SCA_GPIO.GPIO_W_DATAOUT, currently_set)
        return self.read_gpio(line)  # in order to check it is actually set

    @gpio_byname
    def set_gpio_direction(self, line, to=1):
        self.enable_gpio()  # enable GPIO
        currently_set = self.rw_reg(SCA_GPIO.GPIO_R_DIRECTION).value()
        if (currently_set & (1 << line)) and to==0:
            currently_set ^= (1 << line)
        elif to==1:
            currently_set |= (1 << line)
        self.rw_reg(SCA_GPIO.GPIO_W_DIRECTION, currently_set)
        currently_set = self.rw_reg(SCA_GPIO.GPIO_R_DIRECTION).value()
        return (currently_set >> line) & 1  # in order to check it is actually set

    def reset_gpio(self):
        self.enable_gpio()  # enable GPIO
        self.rw_reg(SCA_GPIO.GPIO_W_DATAOUT, 0)
        self.rw_reg(SCA_GPIO.GPIO_W_DIRECTION, 0)

    def disable_gpio(self):
        self.configure_control_registers(en_gpio=0)

    def disable_adc(self):
        self.configure_control_registers(en_adc=0)
        self.i2c_enabled = 0

    def config_gpios(self, verbose=False): #read and print all adc values
        gpio_dict = self.gpio_mapping
        if verbose:
            print("Configuring SCA GPIO Pins...")
        for gpio_reg in gpio_dict.keys():
            pin         = gpio_dict[gpio_reg]['pin']
            direction   = int(gpio_dict[gpio_reg]['direction'] == 'out')
            comment     = gpio_dict[gpio_reg]['comment']
            default     = gpio_dict[gpio_reg]['default']
            if self.verbose:
                print("Setting SCA GPIO pin %s (%s) to %s with value %s"%(pin, comment, gpio_dict[gpio_reg]['direction'], default))
            self.set_gpio(pin, default)  # NOTE this is important because otherwise the GPIO pin can be set to a false default value when switched to output
            self.set_gpio_direction(pin, direction)
            if not self.read_gpio(pin) == default:
                self.set_gpio(pin, default)  # redundant but keep it

    def get_I2C_channel(self, channel):
        channel_str = hex(channel).upper()[-1]
        if channel < 16:
            return SCA_CRB.ENI2C0 + channel  # offset in channel count
        else:
            raise RuntimeError(f"SCA only has 16 I2C channels, don't know what to do with channel {channel}")

    def I2C_write(self, reg=0x0, val=0x0, master=3, slave_addr=0x48, adr_nbytes=2, freq=2):
        # wrapper function to have similar interface as lpGBT I2C_write
        self.enable_I2C(channel=master)
        adr_bytes = [ ((reg >> (8*i)) & 0xff) for i in range(adr_nbytes) ]
        if isinstance(val, int):
            data_bytes = [val]
        elif isinstance(val, list):
            data_bytes = val
        else:
            raise("data must be an int or list of ints")
        self.I2C_write_multi(adr_bytes + data_bytes, channel=master, servant=slave_addr, freq=freq)

    def I2C_read(self, reg=0x0, master=3, slave_addr=0x48, nbytes=1, adr_nbytes=2, freq=2, timeout=0.1):
        # wrapper function to have similar interface as lpGBT I2C_read
        if nbytes > 1 or adr_nbytes>1:
            start_time = time.time()
            while True:
                try:
                    return self.I2C_read_multi(channel=master, servant=slave_addr, reg=reg, nbytes=nbytes, adr_nbytes=adr_nbytes, freq=freq)
                except:
                    if (time.time() - start_time) < timeout:
                        pass
                    else:
                        #print("I2C_read in SCA timed out.")  # not printing this. SCA will time out e.g. if we're trying to see if a module/ETROC is connected on an empty slot!
                        raise RuntimeError("SCA timed out")
        else:
            return self.I2C_read_single_byte(channel=master, servant=slave_addr, reg=reg, freq=freq)

    def I2C_read_single_byte(self, channel=3, servant=0x48, reg=0x00, freq=2):
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)

        # write to the pointer reg
        self.I2C_write_single_byte(channel=channel, servant=servant, data=reg)

        # single byte read
        res = self.rw_cmd(
            SCA_I2C.I2C_S_7B_R,
            self.get_I2C_channel(channel),
            servant<<24,
            0x0,
        ).value()

        status = (res >> 24)
        success = (status & 4)
        if success:
            return (res >> 16) & 255
        else:
            #print ("Read not successful")
            return False

    def I2C_write_single_byte(self, channel, servant, data, freq=2):
        #enable channel
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)
        #single byte write
        data_field = (servant<<24) | ((data & 255) << 16) #[31:24] is servant address, [23:16] is data byte
        res = self.rw_cmd(
            SCA_I2C.I2C_S_7B_W,
            self.get_I2C_channel(channel),
            data_field,
        ).value()
        status = res >> 24
        success = status & 4
        if success:
            return True
        else:
            raise RuntimeError(f"I2C write not successful, status={status}")

    def I2C_write_ctrl(self, channel, data):
        #enable channel
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)
        #write control register
        data_field = data << 24
        res = self.rw_cmd(SCA_I2C.I2C_W_CTRL, self.get_I2C_channel(channel), data_field).value()
        return #no status register to return?

    def I2C_read_ctrl(self, channel):
        #enable channel
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)
        #read control register
        res = self.rw_cmd(SCA_I2C.I2C_R_CTRL, self.get_I2C_channel(channel), 0x0).value()
        return res >> 24

    def I2C_read_multi(self, channel=3, servant=0x48, nbytes=1, reg=0x0, adr_nbytes=2, freq=2):
        adr_bytes = [ ((reg >> (8*i)) & 0xff) for i in range(adr_nbytes) ]
        #enable channel
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)

        #configure NBYTES in the control register
        self.I2C_write_ctrl(channel, nbytes<<2 | freq)
        # write to the pointer reg
        self.I2C_write_multi(adr_bytes, channel, servant)
        #self.I2C_write_single_byte(channel=channel, servant=servant, data=reg)
        #multi-byte read
        start_time = time.time()
        cmd_res = self.rw_cmd(SCA_I2C.I2C_M_7B_R, self.get_I2C_channel(channel), (servant<<24)).value()
        status = cmd_res >> 24
        success = status & 4
        while not success:
            cmd_res = self.rw_cmd(SCA_I2C.I2C_M_7B_R, self.get_I2C_channel(channel), (servant<<24)).value()
            status = cmd_res >> 24
            success = status & 4
            if time.time() - start_time > 0.1:
                raise TimeoutError("I2C_M_7B_R not successful, status = {}".format(status))

        #read data register
        #we are counting backwards because we need to call I2C_R_DATA3 to get data bytes 0, 1, 2, 3, and so on.
        #[I2C_R_DATA3, I2C_R_DATA2, I2C_R_DATA1, I2C_R_DATA0]
        data_registers = [SCA_I2C.I2C_R_DATA3 - SCA_I2C.I2C_RW_DATA_OFFSET * n for n in range(((nbytes-1)//4) + 1)]
        out_bytes = []

        for page in range((((nbytes-1)//4) + 1)):
            page_value = self.rw_cmd(data_registers[page], self.get_I2C_channel(channel), 0x0).value() #execute I2C_R_DATA[3,2,1,0]
            for byte in range(4):
                if (byte + 4*page) < nbytes:
                    mask = 255 << (8 * (3 - byte))
                    return_byte = (page_value & mask) >> (8 * (3 - byte))
                    out_bytes.append(return_byte)

        if len(out_bytes) > 1:
            return out_bytes
        else:
            return out_bytes[0]

    def I2C_write_multi(self, data, channel=3, servant=0x48, freq=2):
        if not type(data) == list:
            data = [data]
        nbytes = len(data)
        #enable channel
        if (self.i2c_enabled & (1<<channel)) == 0:
            self.enable_I2C(channel=channel)
        #configure NBYTES in the control register
        self.I2C_write_ctrl(channel, nbytes<<2 | freq)
        #begin writing to the data registers [I2C_W_DATA0, I2C_W_DATA1, I2C_W_DATA2, I2C_W_DATA3]
        data_registers = [SCA_I2C.I2C_W_DATA0 + SCA_I2C.I2C_RW_DATA_OFFSET * n for n in range(((nbytes-1)//4) + 1)]
        for page in range(((nbytes-1)//4) + 1):
            cmd_val = 0x0
            for byte in range(4):
                if (byte + (4 * page)) < nbytes:
                    write_byte = data[byte + (4*page)] << (8 * (3 - byte))
                    cmd_val = cmd_val + write_byte #append the data byte to the correct position in the command value
            self.rw_cmd(data_registers[page], self.get_I2C_channel(channel), cmd_val)

        #once data registers are filled, execute I2C_M_7B_W
        start_time = time.time()
        data_field = (servant<<24)
        cmd_res = self.rw_cmd(SCA_I2C.I2C_M_7B_W, self.get_I2C_channel(channel), data_field).value()
        status = cmd_res >> 24
        success = status & 4
        while not success:
            cmd_res = self.rw_cmd(SCA_I2C.I2C_M_7B_W, self.get_I2C_channel(channel), data_field).value()
            status = cmd_res >> 24
            success = status & 4
            if time.time() - start_time > 0.3:
                raise TimeoutError("I2C_M_7B_W not successful, status = {}".format(status))


    def I2C_status(self, channel=3, verbose=1):
        # returns whether last transaction was successful
        self.enable_I2C(channel=channel)
        res = self.rw_cmd(SCA_I2C.I2C_R_STR, self.get_I2C_channel(channel), 0x0, 0x0).value()
        status = (res >> 24)
        success = (status & (1<<2)) >> 2 # bit 2 is for success
        if success:
            if verbose:
                print ("Last transaction successful!")
            else:
                pass
        else:
            print ("Last transaction not successful!")
        if (status & (1<<3)):
            print ("SDA/I2C bus broken")
        if (status & (1<<5)):
            print ("Invalid command")
        if (status & (1<<6)):
            print ("Operation not acknowledged by servant")

        return success

    def read_temp_i2c(self, channel=3):
        res = self.I2C_read_multi(channel=channel, servant = 0x48, nbytes=2)
        temp_dig = (res[0] << 4) + (res[1] >> 4)
        return temp_dig*0.0625

