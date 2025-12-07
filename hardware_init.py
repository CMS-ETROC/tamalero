import sys
import os
import time
from tamalero.FIFO import FIFO
from tamalero.ETROC import ETROC
from tamalero.colors import green, red, yellow
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.KCU import KCU

# We import the config type hint only for the IDE, to help with auto-complete
from settings import DAQConfig

class ETROCSystem:
    def __init__(self, config: DAQConfig):
        self.cfg = config
        self.kcu = None
        self.rb = None
        self.etroc_chips = []
        self.connected_names = []  # To track which specific chips succeeded

    def connect(self):
        """Master function to connect to everything in order."""
        print('--- HARDWARE INITIALIZATION ---')
        self._init_kcu()
        self._init_readout_board()
        self._init_chips()
        return self

    def _init_kcu(self):
        print("1. Connecting to KCU...")
        ipb_path = f"chtcp-2.0://localhost:10203?target={self.cfg.kcu_ip}:50001"
        # Assuming environment variable is set, otherwise hardcode path or add to config
        generic_xml_path = os.path.expandvars("$TAMALERO_BASE/address_table/generic/etl_test_fw.xml")

        self.kcu = KCU(
            name="kcu",
            ipb_path=ipb_path,
            adr_table=generic_xml_path
        )

        # Quick Loopback Test
        self.kcu.write_node("LOOPBACK.LOOPBACK", 0xABCD1234)
        if self.kcu.read_node("LOOPBACK.LOOPBACK").value() == 0xABCD1234:
            print(green("   KCU Loopback test PASSED"))
        else:
            print(red("   KCU Loopback test FAILED"))

    def _init_readout_board(self):
        print("2. Initializing Readout Board...")
        self.rb = ReadoutBoard(
            rb=self.cfg.readout_board_id,
            kcu=self.kcu,
            config=self.cfg.readout_board_config,
            trigger=False,
            verbose=False
        )
        print(green(f"   Readout Board version: {self.rb.ver}"))

    def _init_chips(self):
        print("3. Initializing ETROC chips...")
        self.etroc_chips = []
        self.connected_names = []

        for i, addr in enumerate(self.cfg.etroc_addresses):
            name = self.cfg.etroc_names[i]
            print(f"   Attempting {name} (0x{addr:02X})...", end=" ")

            try:
                etroc = ETROC(
                    self.rb,
                    master='lpgbt',
                    i2c_adr=addr,
                    i2c_channel=1,
                    elinks=self.cfg.etroc_elinks_map,
                    strict=False,
                    verbose=False
                )
                self.etroc_chips.append(etroc)

                if etroc.is_connected():
                    self.connected_names.append(name)
                    print(green("Connected"))
                    # Optional: Set power mode high immediately on connect
                    etroc.set_power_mode(mode='high', row=0, col=0, broadcast=True)
                else:
                    self.connected_names.append(None)
                    print(red("Not Responding"))

            except Exception as e:
                print(red(f"Error: {e}"))
                self.etroc_chips.append(None)
                self.connected_names.append(None)

    def configure_trigger(self):
        """Applies trigger configuration from Config object"""
        print("\n4. Configuring Trigger System...")

        print(f"    Trigger mask: {self.cfg.trigger_enable_mask}")
        print(f"    Trigger bit size: {self.cfg.trigger_data_size}")
        print(f"    Trigger delay: {self.cfg.trigger_delay_sel}\n")

        # Write trigger settings
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_ENABLE_MASK", self.cfg.trigger_enable_mask)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_DATA_SIZE", self.cfg.trigger_data_size)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_DLY_SEL", self.cfg.trigger_delay_sel)
        time.sleep(0.1)

        # Verify Elink Locks
        all_locked = True
        for elink in [0, 4, 8, 12]:
            if not self._ensure_lock(elink):
                all_locked = False

        if not all_locked:
            print(red("FATAL: Some E-links failed to lock."))
            sys.exit(1)

        print(green("Trigger system ready."))

    def _ensure_lock(self, elink, max_retries=5):
        """Internal helper to retry locking"""
        for i in range(max_retries):
            if self.rb.etroc_locked(elink, slave=False):
                print(green(f"   E-link {elink} locked status: locked"))
                return True
            print(yellow(f"   E-link {elink} not locked, retrying bitslip ({i+1}/{max_retries})..."))
            self.rb.rerun_bitslip()
            time.sleep(0.5)
        return False
