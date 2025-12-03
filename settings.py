from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DAQConfig:
    # Hardware Settings
    kcu_ip: str = "192.168.0.10"
    readout_board_id: int = 0
    readout_board_config: str = 'default'

    # ETROC Settings
    etroc_addresses: List[int] = field(default_factory=lambda: [0x60, 0x61, 0x62, 0x63])
    etroc_names: List[str] = field(default_factory=lambda: ['ET2p02_PT_IH2', 'ET2p02_PT_IH3', 'ET2p02_PT_IH5', 'ET2p02_PT_IH24'])
    etroc_elinks_map: Dict[int, List[int]] = field(default_factory=lambda: {0: [0, 4, 8, 12]})

    # Thresholds
    th_offsets: Dict[str, int] = field(default_factory=lambda: {
        'ET2p02_PT_IH2': 20, 'ET2p02_PT_IH3': 20,
        'ET2p02_PT_IH5': 20, 'ET2p02_PT_IH24': 20
    })

    # Trigger Settings
    trigger_enable_mask: int = 0x1
    trigger_data_size: int = 1
    trigger_delay_sel: int = 470

    # File/Path Settings
    path_to_figure: str = '/home/daq/ETROC2_KCU105/ETROC-figures'
    path_to_hist: str = '/home/daq/ETROC2_KCU105/ETROC-History'
    chunk_size: int = 500
    max_file_size_bytes: int = 120 * 1024 * 1024

    # Dimensions
    pixel_row: int = 16
    pixel_col: int = 16