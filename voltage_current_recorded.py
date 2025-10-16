import os
import sys
import time
import yaml
from datetime import datetime
import argparse
from tamalero.colors import green, red, yellow
from tamalero.KCU import KCU
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import get_kcu

KCU_IP = '192.168.0.10'
READOUT_BOARD_ID = 0
MONITORING_INTERVAL_SECONDS = 60   ## default time interval
READOUTBOARD_CONFIG = 'default'
CONFIG_FILE_PATH = 'configs/rb_default_v2_smu.yaml'
LOG_FILE = 'voltage&current_record.csv'

def parse_argument():
    parser = argparse.ArgumentParser(description='ETROC Current and Voltage Recorder Script')
    
    parser.add_argument('-t', '--time-interval', 
                        type=float, 
                        default=MONITORING_INTERVAL_SECONDS,
                        help='Time interval between measurements in seconds (default: %(default)s)')
    
    parser.add_argument('-o', '--output', 
                        type=str, 
                        default=LOG_FILE,
                        help='Output CSV file name (default: %(default)s)')
    
    parser.add_argument('-c', '--config', 
                        type=str, 
                        default=CONFIG_FILE_PATH,
                        help='Configuration file path (default: %(default)s)')
    
    parser.add_argument('-ip', '--kcu-ip', 
                        type=str, 
                        default=KCU_IP,
                        help='KCU IP address (default: %(default)s)')
    
    parser.add_argument('-rb', '--readout-board', 
                        type=int, 
                        default=READOUT_BOARD_ID,
                        help='Readout board ID (default: %(default)s)')
    
    args = parser.parse_args()
    
    if not 1 <= args.time_interval <= 255:
        parser.error(f"Time interval must be between 1 and 255 seconds. Got: {args.time_interval}")
    
    return args

def main():
    args = parse_argument()
    
    print("--- ETROC All Voltage Recorder Script ---")
    print(f"Time interval: {args.time_interval} seconds")
    print(f"Output file: {args.output}")
    print(f"Config file: {args.config}")
    print(f"KCU IP: {args.kcu_ip}")
    print(f"Readout Board ID: {args.readout_board}")
    print("-" * 40)

    try:
        print(f"Connecting to KCU with ip adr: {args.kcu_ip}...")
        ipb_path = f'chtcp-2.0://localhost:10203?target={args.kcu_ip}:50001'
        generic_path = os.path.expandvars('$TAMALERO_BASE/address_table/generic/etl_test_fw.xml')

        kcu = KCU(
            name = 'kcu',
            ipb_path = ipb_path,
            adr_table = generic_path
        )
        print(f"Initializing Readout Board #{args.readout_board}...")
        
        rb = ReadoutBoard(
            rb=args.readout_board,
            kcu=kcu,
            config=READOUTBOARD_CONFIG, 
            trigger=True,     
            verbose=False,
            allow_bad_links=True
        )
        print(green(f"Readout Board version detected: {rb.ver}"))
        
        if rb.trigger:
            print(green("Found Trigger LPGBT and initialized"))
        else:
            print(yellow("Warning: Could not find Trigger LPGBT and can read it channels"))

    except Exception as e:
        print(f"\nError when initializing hardware: {e}")
        sys.exit(1)

   
    try:
        # print("\n--- ADC Channel Debug ---")
        # for i in range(8):
        #     raw = rb.DAQ_LPGBT.read_adc_raw(i)
        #     print(f"ADC{i}: raw={raw}, voltage={raw/1023.0:.3f}V")
        # print("--- End Debug ---\n")
        
        # pos = rb.DAQ_LPGBT.read_adc_raw(2)
        # neg = rb.DAQ_LPGBT.read_adc_raw(3)
        # print(f"ADC2: {pos}, ADC3: {neg}, Diff: {pos-neg}")

        # rb.DAQ_LPGBT.get_current_dac_status(channel=2, summary=True)

        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        adc_channels = []
        lpgbt_adc_map = config_data.get('LPGBT', {}).get('adc', {})
        
        for name, properties in lpgbt_adc_map.items():
            lpgbt_type = properties.get('lpgbt', 'daq')
            is_differential = properties.get('differential', False)

            if is_differential and 'pin_pos' in properties and 'pin_neg' in properties:
                 adc_channels.append({
                    'name': name,
                    'type': 'differential',
                    'pin_pos': properties['pin_pos'],
                    'pin_neg': properties['pin_neg'],
                    'conv': properties.get('conv', 1.0),
                    'lpgbt': lpgbt_type
                })
            elif not is_differential and 'pin' in properties:
                adc_channels.append({
                    'name': name,
                    'type': 'single_ended',
                    'pin': properties['pin'],
                    'conv': properties.get('conv', 1.0),
                    'lpgbt': lpgbt_type
                })

        if not adc_channels:
            print(f"Error: Could not find any valid adc channels in config file {args.config}")
            return
        
        print("\n Those LPGBT adc Channels are being monitored:")
        for ch in adc_channels:
            print(f" -{ch['name']} (LPGBT: {ch['lpgbt']}, Type: {ch['type']})")

    except FileNotFoundError:
        print(f"Error: Can't find config file '{args.config}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error when reading config file: {e}")
        sys.exit(1)

    print(f"\nStart to get vtemp data every {args.time_interval} seconds。 Press Ctrl+C to stop.")
    
    output_file = None
    if args.output:
        try:
            output_file = open(args.output, 'w')
            header = "timestamp," + ",".join([f"{ch['name']}_raw,{ch['name']}_volts" for ch in adc_channels])
            output_file.write(header + "\n")
            print(f"Vtemp data will be saved into {args.output}")
        except IOError as e:
            print(f"Error: Can't open file {args.output}: {e}")
            output_file = None

    try:
        while True:
            readings = {}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for channel in adc_channels:
                name = channel['name']
                lpgbt_obj = None

                if channel['lpgbt'] == 'trigger':
                    if rb.trigger:
                        lpgbt_obj = rb.TRIG_LPGBT
                    else:
                        readings[name] = {'raw': 'N/A', 'value': 'N/A'}
                        continue
                else:
                    lpgbt_obj = rb.DAQ_LPGBT
                try:
                    if channel['type'] == 'single_ended':
                        raw_value = lpgbt_obj.read_adc(channel['pin'], calibrate=False)
                        voltage = (raw_value / 1023.0) * channel['conv']
                    else: # differential
    
                        raw_pos = lpgbt_obj.read_adc(channel['pin_pos'], calibrate=False)
                        raw_neg = lpgbt_obj.read_adc(channel['pin_neg'], calibrate=False)
                        raw_value = raw_pos - raw_neg
                        voltage = (raw_value / 1023.0) * channel['conv']

                    readings[name] = {'raw': raw_value, 'value': voltage}

                except Exception as e:
                    print(f"\Error when reading from {name}: {e}")
                    readings[name] = {'raw': 'ERROR', 'value': 'ERROR'}

            print(f"\n--- {timestamp} ---")
            for ch in adc_channels:
                name = ch['name']
                raw = readings[name]['raw']
                value = readings[name]['value']
                unit = 'A' if 'current' in name else 'V'

                if isinstance(value, float):
                    print(f"{name+':':<20} Raw = {raw:<6} | Value = {value:.4f} {unit}")
                    # print(f"{name+':':<20} Raw = {raw:<6} | Value = {volts:.4f} V")
                else:
                    print(f"{name+':':<20} Raw = {raw:<6} | Value = {value}")
  
            if output_file:
                log_data = []
                for ch in adc_channels:
                    raw_val = readings[ch['name']]['raw']
                    val = readings[ch['name']]['value']
                    if isinstance(val, float):
                        log_data.append(f"{raw_val},{val:.4f}")
                    else:
                        log_data.append(f"{raw_val},{val}")
                output_file.write(f"{timestamp},{','.join(log_data)}\n")
                output_file.flush()

            time.sleep(args.time_interval)

            
    except KeyboardInterrupt:
        print("\n\nStopping get vtemp data。")
    finally:
        if output_file:
            output_file.close()
            print("file saved and closed")

if __name__ == "__main__":
    main()