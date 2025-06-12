"""
Control board class (KCU105). Depends on uhal.
"""
try:
    import uhal
except ModuleNotFoundError:
    print("Running without uhal (ipbus not installed with correct python bindings)")
from tamalero.colors import red, green
import time

class KCU:

    def __init__(self,
                 name="my_device",
                 ipb_path="ipbusudp-2.0://192.168.0.10:50001",
                 adr_table="../module_test_fw/address_tables/etl_test_fw.xml",
                 dummy=False):

        uhal.disableLogging()
        self.auto_dispatch = True  # default -> True

        self.dummy = dummy

        self.max_retries = 20
        if not self.dummy:
            try:
                self.hw = uhal.getDevice("my_device", ipb_path, "file://" + adr_table)
            except:
                raise Exception("uhal can't get device at"+adr_table)
            self.firmware_version = self.get_firmware_version(string=False, verbose=False)
        else:
            self.hw = None
        self.readout_boards = []

    def toggle_dispatch(self):
        self.auto_dispatch = False

    def dispatch(self):
        i = 0
        while i<self.max_retries:
            try:
                self.hw.dispatch()
                self.auto_dispatch = True
                break
            except:
                if i > (self.max_retries-2):
                    raise
                i+=1

    def write_node(self, id, value):
        reg = self.hw.getNode(id)
        if (reg.getPermission() == uhal.NodePermission.WRITE):
            self.action_reg(reg)
        else:
            reg.write(value)
            if self.auto_dispatch:
                self.dispatch()

    def rd_lpgbt_adr(self, rb=0):
        '''
        function for debugging.
        Might be deleted again because it should not be a member in here
        '''
        self.write_node("READOUT_BOARD_%d.SC.TX_REGISTER_ADDR" % rb, adr)
        self.dispatch()
        self.action("READOUT_BOARD_%d.SC.TX_START_READ" % rb)
        valid = self.read_node("READOUT_BOARD_%d.SC.RX_DATA_VALID" % rb).valid()
        if valid:
            return self.read_node("READOUT_BOARD_%d.SC.RX_DATA_FROM_GBTX" % rb)
        print("LpGBT read failed!")
        return None

    def read_node(self, id):
        try:
            reg = self.hw.getNode(id)
        except:
            raise Exception(f"Failed finding node {id} in read_node")
        ret = reg.read()
        if self.auto_dispatch:
            self.dispatch()
        return ret

    def action_reg(self, reg):
        addr = reg.getAddress()
        mask = reg.getMask()
        self.hw.getClient().write(addr, mask)
        if self.auto_dispatch:
            self.dispatch()

    def action(self, id):
        reg = self.hw.getNode(id)
        self.action_reg(reg)

    def print_regs(self):

        for id in self.hw.getNodes():
            reg = self.hw.getNode(id)
            # if (reg.getModule() == ""):
            if (reg.getMode() != uhal.BlockReadWriteMode.HIERARCHICAL):
                print(self.format_reg(reg.getAddress(), reg.getPath()[4:], -1,
                                self.format_permission(reg.getPermission())))

    def get_firmware_version(self, verbose=False, string=True):

        nodes = ("FW_INFO.HOG_INFO.GLOBAL_DATE",
                 "FW_INFO.HOG_INFO.GLOBAL_TIME",
                 "FW_INFO.HOG_INFO.GLOBAL_VER",
                 "FW_INFO.HOG_INFO.GLOBAL_SHA",)

        (date, time, ver, sha) = (map (lambda x : self.read_node(x).value(), nodes))

        if verbose:
            print("Firmware version: %04x/%02x/%02x %02x:%02x:%02x v%x.%x.%x sha=%07x" % (
                date & 0xffff,
                (date >> 16) & 0xff,
                (date >> 24) & 0xff,
                time & 0xff,
                (time >> 8) & 0xff,
                (time >> 8) & 0xff,
                (ver >> 24) & 0xff,
                (ver >> 16) & 0xff,
                (ver >> 0) & 0xff,
                sha))

        res = ver
        if string:
            return "%s.%s.%s"%(res >> 24, (res >> 16) & 0xFF, res & 0xFFFF)
        else:
            return {"major": res >> 24, "minor": (res >> 16) & 0xFF, "patch": res & 0xFFFF}

    def get_firmware_sha(self):
        res = self.read_node("FW_INFO.HOG_INFO.GLOBAL_SHA")
        return hex(res).strip('0x0')

    def get_xml_sha(self):
        res = self.read_node("FW_INFO.HOG_INFO.XML_SHA")
        return hex(res).strip('0x0')

    def get_serial(self):
        # placeholder
        return "000000"

    def status(self):
        print("LPGBT Link Status from KCU:")
        for id in self.hw.getNodes(".*LPGBT.*DOWNLINK.*READY"):
            self.print_reg(self.hw.getNode(id), use_color=True, threshold=1)
        for id in self.hw.getNodes(".*LPGBT.*UPLINK_0.*READY"):
            self.print_reg(self.hw.getNode(id), use_color=True, threshold=1)
        for id in self.hw.getNodes(".*LPGBT.*UPLINK_0.*FEC_ERR_CNT"):
            self.print_reg(self.hw.getNode(id), use_color=True, threshold=1, invert=True)
        for id in self.hw.getNodes(".*LPGBT.*UPLINK_1.*READY"):
            self.print_reg(self.hw.getNode(id), use_color=True, threshold=1)
        for id in self.hw.getNodes(".*LPGBT.*UPLINK_1.*FEC_ERR_CNT"):
            self.print_reg(self.hw.getNode(id), use_color=True, threshold=1, invert=True)

        self.check_clock_frequencies()
        
        for rb in self.readout_boards:
            print(f'Checking Readout Board {rb.rb}')
            locked = self.read_node(f"READOUT_BOARD_{rb.rb}.ETROC_LOCKED").value()
            locked_slave = self.read_node(f"READOUT_BOARD_{rb.rb}.ETROC_LOCKED_SLAVE").value()

            for l in range(28):
                if (locked >> l) & 1:
                    print(green(f'Master elink {l} is locked.'))
            for l in range(28):
                if (locked_slave >> l) & 1:
                    print(green(f'Slave elink {l} is locked.'))

            if locked | locked_slave == 0:
                print(red('Warning: No elink is locked.'))
            print()

    def reset_FEC_error_count(self):
        for i in range(5):
            self.write_node("READOUT_BOARD_%s.LPGBT.FEC_ERR_RESET" % i, 0x1)


    def print_reg(self, reg, threshold=1, maxval=0xFFFFFFFF, use_color=False, invert=False):
        from tamalero.colors import green, red, dummy
        val = reg.read()
        id = reg.getPath()
        self.dispatch()
        if use_color:
            if invert:
                colored = green if (val < threshold and val < maxval) else red
            else:
                colored = green if (val >= threshold and val < maxval) else red
        else:
            colored = dummy
        print(colored(self.format_reg(reg.getAddress(), id[4:], val,
                              self.format_permission(reg.getPermission()))))

    def format_reg(self, address, name, val, permission=""):
        s = "{:<8}{:<8}{:<50}".format("0x%04X" % address, permission, name)
        if (val != -1):
            s = s + "{:<8}".format("0x%08X" % val)
        return s

    def format_permission(self, perm):
        if perm == uhal.NodePermission.READ:
            return "r"
        if perm == uhal.NodePermission.READWRITE:
            return "rw"
        if perm == uhal.NodePermission.WRITE:
            return "w"

    def check_clock_frequencies(self, verbose=False):
        clocks = (('FW_INFO.CLK125_FREQ', 125000000),
                  ('FW_INFO.CLK320_FREQ', 320640000),
                  ('FW_INFO.CLK_40_FREQ',  40080000),
                  ('FW_INFO.REFCLK_FREQ', 320640000),
                  ('FW_INFO.RXCLK0_FREQ', 320640000),
                  ('FW_INFO.RXCLK1_FREQ', 320640000),
                  ('FW_INFO.TXCLK0_FREQ', 320640000),
                  ('FW_INFO.TXCLK1_FREQ', 320640000))

        # freq = int(rd) / 1000000.0
        # print("%s = %6.2f MHz" % (id, freq))

        errs = 0
        tolerance = 3000  # increased tolerance to 3.0kHz (from 2kHz)
        for clock in clocks:
            freq = self.read_node(clock[0]).value()
            expect = clock[1]
            err = freq > expect + tolerance or freq < expect-tolerance
            errs = errs + err
            if (err or verbose):
                self.print_reg(self.hw.getNode(clock[0]), use_color=True, threshold=clock[1] - tolerance, maxval=clock[1] + tolerance)

        return errs

    def send_l1a(self, count=1, quiet=True, max_rate=-1):
        sleeper = 1./max_rate if max_rate>0 else 0
        start_time = time.time()
        for i in range(count):
            time.sleep(sleeper)
            try:
                self.write_node("SYSTEM.L1A_PULSE", 1)
            except:
                print("Couldn't send pulse.")
        timediff = time.time() - start_time
        rate = count/timediff
        if not quiet:
            print(f"Sent {count} L1As at a rate of {rate}Hz")
        return rate
