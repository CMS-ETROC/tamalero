import time
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# This matches any file ending in .dat OR the exact yaml filename.
ALLOWED_SUFFIXES = (".dat")

# ---------------------
class DAQTransferHandler(FileSystemEventHandler):
    def __init__(self, args):
        """Store the arguments passed from argparse"""
        super().__init__()
        self.args = args

    def on_closed(self, event):
        """
        Triggered natively by Ubuntu's inotify when a file is closed.
        """
        if not event.is_directory and event.src_path.endswith(ALLOWED_SUFFIXES):
            # Convert the string path from watchdog into a Path object
            filepath = Path(event.src_path)

            print(f"[{time.strftime('%X')}] DAQ closed file safely: {filepath.name}")
            self.transfer_file(filepath)

    def transfer_file(self, filepath):
        """
        Transfers the file using rsync.
        """
        # Ensure filepath is a Path object just in case it's called directly elsewhere
        filepath = Path(filepath)

        remote_path = f"{self.args.user}@{self.args.server}:{self.args.remote_dir}/"

        print(f"Starting transfer for {filepath.name}...")

        try:
            # subprocess.run accepts Path objects directly for the file path
            subprocess.run(
                ["rsync", "-azq", filepath, remote_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[{time.strftime('%X')}] Success: {filepath.name} copied to remote server.")

        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%X')}] Error transferring {filepath.name}: {e.stderr}")

# ---------------------
def main(args):
    # Convert input directory string to a Path object
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    # --- NEW: INITIAL SYNC for run metadata yaml---
    print(f"Checking for existing files to sync before monitoring...")
    for filepath in input_dir.iterdir():
        if filepath.is_file() and filepath.name.endswith("run_metadata.yaml"):
            remote_path = f"{args.user}@{args.server}:{args.remote_dir}/"
            print(f"Initial sync for {filepath.name}...")
            try:
                subprocess.run(
                    ["rsync", "-azq", "--mkpath", filepath, remote_path],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed initial sync for {filepath.name}: {e}")
    # -------------------------

    # Pass the args into your handler
    event_handler = DAQTransferHandler(args)
    observer = Observer()

    # watchdog expects a string for the scheduling path, so we cast it back to str
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()

    print(f"Monitoring {input_dir} for closed .dat and metadata files...")
    print(f"Target: {args.user}@{args.server}:{args.remote_dir}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        observer.stop()

    observer.join()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Copy data in real time')
    parser.add_argument('-d', '--input-dir', type=str, required=True, dest='input_dir', help='Directory to watch')
    parser.add_argument('--user', type=str, required=True, dest='user', help='Username of remote server')
    parser.add_argument('--server', type=str, required=True, dest='server', help='Remote server name')
    parser.add_argument('-r', '--remote-dir', type=str, required=True, dest='remote_dir', help='Path to directory in remote server')
    args = parser.parse_args()

    print(f"Remote path: {args.user}@{args.server}:{args.remote_dir}/")

    main(args)