import os
import sys
import time
import yaml
import argparse
from datetime import datetime, timezone
from tamalero.colors import green, red, yellow

from tamalero.KCU import KCU
from tamalero.ReadoutBoard import ReadoutBoard

KCU_IP = '192.168.0.10'
READOUT_BOARD_ID = 0
MONITORING_INTERVAL_SECONDS = 5.0
READOUTBOARD_CONFIG = 'default'
CONFIG_FILE_PATH = 'configs/rb_default_v2_smu.yaml'

def main(args):
    print("--- ETROC VTEMP script ---")

    try:
        print(f"Connecting to KCU with ip adr: {KCU_IP}...")
        ipb_path = f'chtcp-2.0://localhost:10203?target={KCU_IP}:50001'
        generic_path = os.path.expandvars('$TAMALERO_BASE/address_table/generic/etl_test_fw.xml')

        kcu = KCU(
            name = 'kcu',
            ipb_path = ipb_path,
            adr_table = generic_path
        )
        print(f"Initializing Readout Board #{READOUT_BOARD_ID}...")

        rb = ReadoutBoard(
            rb=READOUT_BOARD_ID,
            kcu=kcu,
            config=READOUTBOARD_CONFIG,
            trigger=False,
            verbose=False
        )
        print(green(f"Readout Board version detected: {rb.ver}"))

    except Exception as e:
        print(f"\nError when initializing hardware: {e}")
        sys.exit(1)

    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config_data = yaml.safe_load(f)

        vtemp_channels = []
        lpgbt_adc_map = config_data.get('LPGBT', {}).get('adc', {})

        for name, properties in lpgbt_adc_map.items():
            if 'VTEMP' in name.upper():
                if 'pin' in properties:
                    vtemp_channels.append({
                        'name': name,
                        'pin': properties['pin']
                    })

        if not vtemp_channels:
            print(f"Error: Can't find VTEMP channal in config file {CONFIG_FILE_PATH}")
            return

        print("\nFind VTEMP channels:")
        for ch in vtemp_channels:
            print(f"  - {ch['name']} (LPGBT ADC Pin: {ch['pin']})")

    except FileNotFoundError:
        print(f"Error: Can't find config file '{CONFIG_FILE_PATH}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error when reading config file: {e}")
        sys.exit(1)

    print(f"\nStart to get vtemp data every {MONITORING_INTERVAL_SECONDS} seconds。 Press Ctrl+C to stop.")

    output_file = None
    output_file_name = f'{args.output}.csv'
    try:
        output_file = open(output_file_name, 'w')
        header = "timestamp," + ",".join([f"{ch['name']}_raw,{ch['name']}_volts" for ch in vtemp_channels])
        output_file.write(header + "\n")
        print(f"Vtemp data will be saved into {output_file_name}")
    except IOError as e:
        print(f"Error: Can't open file {output_file_name}: {e}")
        output_file = None

    try:
        while True:
            readings = {}
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            for channel in vtemp_channels:
                try:
                    raw_value = rb.DAQ_LPGBT.read_adc(channel['pin'], calibrate=False)
                    voltage = raw_value / 1023.0
                    readings[channel['name']] = {'raw': raw_value, 'volts': voltage}

                except Exception as e:
                    print(f"\nError when reading from {channel['name']}: {e}")
                    readings[channel['name']] = {'raw': 'ERROR', 'volts': 'ERROR'}

            print(f"\n--- {timestamp} ---")
            for ch in vtemp_channels:
                name = ch['name']
                raw = readings[name]['raw']
                volts = readings[name]['volts']
                if isinstance(volts, float):
                    print(f"{name+':':<10} Raw = {raw:<5} | Voltage = {volts:.4f} V")
                else:
                    print(f"{name+':':<10} Raw = {raw:<5} | Voltage = {volts}")

            if output_file:
                log_data = []
                for ch in vtemp_channels:
                    raw_val = readings[ch['name']]['raw']
                    volt_val = readings[ch['name']]['volts']
                    if isinstance(volt_val, float):
                        log_data.append(f"{raw_val},{volt_val:.4f}")
                    else:
                        log_data.append(f"{raw_val},{volt_val}")
                output_file.write(f"{timestamp},{','.join(log_data)}\n")
                output_file.flush()

            time.sleep(MONITORING_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopping get vtemp data。")
    finally:
        if output_file:
            output_file.close()
            print("file saved and closed")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run VTemp Recorder')
    parser.add_argument('-o', '--output', type=str, required=True, dest='output', help='output file name, extension is fixed to csv')
    args = parser.parse_args()

    main(args)