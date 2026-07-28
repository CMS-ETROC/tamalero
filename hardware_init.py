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

    def hardware_reset(self):
        """
        Pulls GPIO pins low then high to power-cycle the I2C bus.
        """
        print(yellow("   [Reset] Pulling LPGBT GPIOs to power-cycle the I2C bus..."))
        # Pull all 4 GPIO control pins low (assert reset)
        for pin in range(4):
            self.rb.DAQ_LPGBT.set_gpio_direction(pin, 1)
            self.rb.DAQ_LPGBT.set_gpio(pin, 0)
        time.sleep(0.2)

        # Release pins high (de-assert reset)
        for pin in range(4):
            self.rb.DAQ_LPGBT.set_gpio(pin, 1)
        time.sleep(0.5)

    def _init_chips(self, max_retries=3):
        print("3. Initializing ETROC chips...")

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(yellow(f"\n   [Reset] Attempt {attempt}/{max_retries}. Missing chips detected, performing hardware reset..."))
                self.hardware_reset()

            temp_etroc_chips = []
            temp_connected_names = []
            all_success = True

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

                    if etroc.is_connected():
                        temp_etroc_chips.append(etroc)
                        temp_connected_names.append(name)
                        print(green("Connected"))
                    else:
                        temp_etroc_chips.append(None)
                        temp_connected_names.append(None)
                        print(red("Not Responding"))
                        all_success = False

                except Exception as e:
                    print(red(f"Error: {e}"))
                    temp_etroc_chips.append(None)
                    temp_connected_names.append(None)
                    all_success = False

            # Update system lists with the results of this attempt
            self.etroc_chips = temp_etroc_chips
            self.connected_names = temp_connected_names

            if all_success:
                print(green("   All chips connected successfully."))
                return

        if not all_success:
            print(red(f"\n   FATAL: Could not connect to all chips after {max_retries} attempts. Halting initialization."))
            sys.exit(1)

    def pll_fc_calibration_and_set_to_high_power_mode(self):
        """Runs PLL/FC calibration and sets power mode for all successfully connected chips."""
        for etroc, name in zip(self.etroc_chips, self.connected_names):

            if etroc is None:
                continue

            # PLL calibration
            etroc.wr_reg("asyPLLReset", 0)
            time.sleep(0.1)
            etroc.wr_reg("asyPLLReset", 1)

            etroc.wr_reg('asyStartCalibration', 0)
            time.sleep(0.1)
            etroc.wr_reg('asyStartCalibration', 1)

            # FC calibration
            etroc.wr_reg('asyAlignFastcommand', 1)
            time.sleep(0.1)
            etroc.wr_reg('asyAlignFastcommand', 0)

            # Global Readout calibration
            etroc.wr_reg('asyResetGlobalReadout', 0)
            time.sleep(0.1)
            etroc.wr_reg('asyResetGlobalReadout', 1)

            # Set power mode high
            etroc.set_power_mode(mode=self.cfg.power_mode, row=0, col=0, broadcast=True)

            print(green(f"\n   {name} PLL/FC calibrated and set to high power mode."))

    def set_gain_mode(self):
        for etroc, name in zip(self.etroc_chips, self.connected_names):

            if etroc is None:
                continue

            # Set power mode high
            etroc.wr_reg('RfSel', self.cfg.gain_mode, row=0, col=0, broadcast=True)
            print(green(f"\n   {name} Gain mode set to {self.cfg.gain_mode}."))

    def check_PS_status(self):
        for etroc, etroc_name in zip(self.etroc_chips, self.connected_names):

            ps_late_array = [etroc.rd_reg('PS_Late') for _ in range(15)]
            new_ps_late_array = "N/A (No Reset)"

            if not any(ps_late_array):
                # Perform Reset Pulse
                etroc.wr_reg('PS_CapRst', 1)
                etroc.wr_reg('PS_CapRst', 0)

                # Re-check status
                new_ps_late_array = [etroc.rd_reg('PS_Late') for _ in range(15)]

            print(f"\n    {'='*10} {etroc_name} {'(PS_Late register)'} {'='*10}")
            print(f"    Before Reset: {ps_late_array}")
            print(f"    After Reset:  {new_ps_late_array}")
            print(f"    {'='*40}")

    def configure_trigger(self):
        """Applies trigger configuration from Config object"""
        print("\n5. Configuring Trigger System...")

        print(f"    Trigger mask: {self.cfg.trigger_enable_mask}")
        print(f"    Trigger bit size: {self.cfg.trigger_data_size}")
        print(f"    Trigger delay: {self.cfg.trigger_delay_sel}")
        print(f"    Trigger logic: {self.cfg.trigger_logic} (0: OR, 1: AND)\n")

        # Write trigger settings
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_ENABLE_MASK", self.cfg.trigger_enable_mask)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_DATA_SIZE", self.cfg.trigger_data_size)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_DLY_SEL", self.cfg.trigger_delay_sel)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.TRIG_COMBINATION_LOGIC", self.cfg.trigger_logic)
        time.sleep(0.1)

        # Try BCR (BC0) Pulse and/or ECR (Reset L1A counter) Pulse
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.BC0_PULSE", 1)
        self.rb.kcu.write_node(f"READOUT_BOARD_{self.rb.rb}.ECR_PULSE", 1)
        time.sleep(0.1)

        # Verify Elink Locks
        all_locked = True
        for elink in self.cfg.etroc_elinks_map[0]:
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
