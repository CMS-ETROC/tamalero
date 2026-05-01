import struct
from pathlib import Path
from datetime import datetime
from settings import DAQConfig

def generate_run_dir(root_path, run_type="beam", note=""):
    """
    Scans the data_root for the next Run number and returns a full Path.
    Format: {root}/run_{N}_{Type}_{Timestamp}_{Note}
    """
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Find the next Run Number
    max_run = 0
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.startswith("run_"):
            try:
                # Expecting format Run_XXX_...
                parts = folder.name.split('_')
                run_num = int(parts[1])
                if run_num > max_run:
                    max_run = run_num
            except (IndexError, ValueError):
                continue

    next_run = max_run + 1

    # 2. Generate Timestamp
    timestamp = datetime.now().strftime("%Y%m%d")

    # 3. Clean the note (remove spaces)
    clean_note = f"{note.replace(' ', '')}" if note else ""

    # 4. Construct Name
    dir_name = f"run_{next_run:03d}_{run_type}_{timestamp}_{clean_note}"
    full_path = root / dir_name

    print(f"Auto-generated Output Directory: {full_path}")
    return full_path

class DataWriter:
    def __init__(self, config: DAQConfig, run_start_time=None):
        self.cfg = config
        self.out_dir = Path(self.cfg.outdir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.file_index = 0
        self.current_file = None
        self.events_in_file = 0
        self.start_time = run_start_time if run_start_time else datetime.now()

        # Open the first file immediately
        self._open_new_file()

    def _open_new_file(self):
        """Internal method to close current and open next file."""
        if self.current_file and not self.current_file.closed:
            self.current_file.close()
            print(f"   [DataWriter] Closed file: {self.current_file.name}")

        filename = self.out_dir / f"file_{self.file_index}_CE.dat"
        self.current_file = open(filename, "wb")
        self.events_in_file = 0
        self.file_index += 1
        print(f"   [DataWriter] Opened: {filename.name}")

    def write(self, raw_data):
        """
        Takes raw FIFO data (list of ints), packs it, and writes to disk.
        Handles automatic file rotation.
        """
        if not raw_data:
            return

        # 1. Pack binary data (Little Endian unsigned int)
        packed_data = struct.pack(f'<{len(raw_data)}I', *raw_data)
        self.current_file.write(packed_data)
        self.events_in_file += 1

        # 2. Check limits (Rotation logic)
        current_size = self.current_file.tell()

        is_full_by_count = (self.events_in_file >= self.cfg.chunk_size)
        is_full_by_size = (current_size >= self.cfg.max_file_size_bytes)

        if is_full_by_count or is_full_by_size:
            self._open_new_file()

    def close(self):
        """Final cleanup."""
        if self.current_file and not self.current_file.closed:
            self.current_file.close()
            print(f"   [DataWriter] Final file closed: {self.current_file.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
