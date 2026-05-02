from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ETROCConfig:
    name: str
    th_offset: int
    elink_id: int
    i2c_id: int
    l1a_delay: int
    is_primary: bool = True ## is_primary is the flag to run calibration. If this set to False, calibration will not run.

@dataclass
class DAQConfig:
    # Hardware Settings
    kcu_ip: str = "192.168.0.10"
    readout_board_id: int = 0
    readout_board_config: str = 'default'

    # --- BOARD DEFINITIONS (The Single Source of Truth) ---
    # Edit ONLY this list to add/remove boards
    boards: List[ETROCConfig] = field(default_factory=lambda: [
        ETROCConfig(name="", i2c_id=0x60, elink_id=0,  th_offset=20, l1a_delay=0x1f5),
        ETROCConfig(name="", i2c_id=0x61, elink_id=4,  th_offset=20, l1a_delay=0x1f5),
        ETROCConfig(name="", i2c_id=0x62, elink_id=8,  th_offset=20, l1a_delay=0x1f5),
        ETROCConfig(name="", i2c_id=0x63, elink_id=12, th_offset=20, l1a_delay=0x1f5),
        # ETROCConfig(name="", i2c_id=0x63, elink_id=12, th_offset=20, l1a_delay=0x1f5, is_primary=False),
    ])

    # --- DERIVED FIELDS (Automatically Calculated) ---
    # We use init=False so the user doesn't have to provide them
    etroc_addresses: List[int] = field(init=False)
    etroc_names: List[str] = field(init=False)
    etroc_elinks_map: Dict[int, List[int]] = field(init=False)
    th_offsets: Dict[str, int] = field(init=False)
    l1a_delays: Dict[str, int] = field(init=False)
    hvs: Dict[str, float] = field(init=False)

    # Trigger Settings
    trigger_enable_mask: int = 0x1 ## 0x11 for dual port
    trigger_data_size: int = 1 ## keep 1
    trigger_delay_sel: int = 469
    trigger_logic:     int = 0 ## 0: OR, 1: AND

    # File/Path Settings
    path_to_figure: str = '/home/daq/ETROC2_KCU105/ETROC-figures'
    path_to_hist: str = '/home/daq/ETROC2_KCU105/ETROC-History'
    chunk_size: int = 1000
    max_file_size_bytes: int = 120 * 1024 * 1024

    # Dimensions
    pixel_row: int = 16
    pixel_col: int = 16

    # Charge injection params
    charge_fc: int = 30
    test_pixels: List[int] = field(default_factory=lambda: [(5, 5), (8, 8)])
    qinj_count: int = 100

    def __post_init__(self):
        """
        Automatically populate the lists and dicts required by the hardware
        drivers based on the 'boards' list above.
        """
        # 1. Extract Lists
        self.etroc_names = [b.name for b in self.boards]
        self.etroc_addresses = [b.i2c_id for b in self.boards]

        # 2. Build Threshold and L1A delay Dictionary
        self.th_offsets = {b.name: b.th_offset for b in self.boards}
        self.l1a_delays = {b.name: b.l1a_delay for b in self.boards}

        # 3. Build Elink Map (assuming key 0 maps to all active elinks)
        all_elinks = [b.elink_id for b in self.boards]
        self.etroc_elinks_map = {0: all_elinks}

        # 4. Pre-populate the HV dictionary with safe defaults (0.0V)
        self.hvs = {b.name: 0.0 for b in self.boards}
