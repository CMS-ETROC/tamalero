from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ETROCConfig:
    name: str
    th_offset: int
    elink_id: int
    i2c_id: int

    def init(name, th_offset, elink_id, i2c_id):
        self.name = name
        self.th_offset = th_offset
        self.elink_id = elink_id
        self.i2c_id = i2c_id

@dataclass
class DAQConfig:
    # Hardware Settings
    kcu_ip: str = "192.168.0.10"
    readout_board_id: int = 0
    readout_board_config: str = 'default'

    boards = [
      ETROCConfig("ET2p02_PT_NH47", 20, 0, 0x60),
      ETROCConfig("ET2p02_PT_IH18", 20, 4, 0x61),
      ETROCConfig("ET2p02_PT_IH21", 20, 8, 0x62),
      ETROCConfig("ET2p02_PT_IH22", 20, 12, 0x63),
    ]

    # ETROC Settings
    etroc_addresses: List[int] = field(default_factory=lambda: [0x60, 0x61, 0x62, 0x63])
    etroc_names: List[str] = field(default_factory=lambda: ['ET2p02_PT_NH47', 'ET2p02_PT_IH18', 'ET2p02_PT_IH21', 'ET2p02_PT_IH22'])
    etroc_elinks_map: Dict[int, List[int]] = field(default_factory=lambda: {0: [0, 4, 8, 12]})

    # Thresholds
    th_offsets: Dict[str, int] = field(default_factory=lambda: {
        'ET2p02_PT_NH47': 20, 'ET2p02_PT_IH18': 20,
        'ET2p02_PT_IH21': 20, 'ET2p02_PT_IH22': 20
    })

    # Trigger Settings
    trigger_enable_mask: int = 0x8
    trigger_data_size: int = 1
    trigger_delay_sel: int = 469

    # File/Path Settings
    path_to_figure: str = '/home/daq/KCU105_NEW/ETROC-figures'
    path_to_hist: str = '/home/daq/KCU105_NEW/ETROC-History'
    chunk_size: int = 1000
    max_file_size_bytes: int = 120 * 1024 * 1024

    # Dimensions
    pixel_row: int = 16
    pixel_col: int = 16

    # Charge injection params
    charge_fc: int = 30
    test_pixels: List[int] = field(default_factory=lambda: [(0, 0), (8, 8)])
    qinj_count: int = 100
