import struct
import gc, time
import numpy as np
import pandas as pd

format_dict = {
    'nbits': 40,
    'bitorder': 'reversed',
    'identifiers': {
        'header': {
            'frame': 0x3C5C000000,
            'mask': 0xFFFFC00000,
        },
        'data': {
          'frame': 0x8000000000,
          'mask': 0x8000000000,
        },
        'filler': {
          'frame': 0x3C5C800000,
          'mask': 0xFFFFC00000,
        },
        'trailer': {
          'frame': 0x0000000000,
          'mask': 0x8000000000,
        },
    },
    'types': ['ea', 'col_id', 'row_id', 'toa', 'cal', 'tot', 'elink', 'full', 'any_full', 'global_full'],
    'data': {
        'header': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40,
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48,
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49,
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50,
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51,
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52,
            },
            'l1counter': {
                'mask': 0x00003FC000,
                'shift': 14,
            },
            'type': {
                'mask': 0x0000003000,
                'shift': 12,
            },
            'bcid': {
                'mask': 0x0000000FFF,
                'shift': 0,
            },
        },
        'data': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'ea': {
                'mask': 0x6000000000,
                'shift': 37
            },
            'col_id': {
                'mask': 0x1E00000000,
                'shift': 33
            },
            'row_id': {
                'mask': 0x01E0000000,
                'shift': 29
            },
            # random test pattern specific
            'col_id2': {
                'mask': 0x001E000000,
                'shift': 25
            },
            'row_id2': {
                'mask': 0x0001E00000,
                'shift': 21
            },
            'bcid': {
                'mask': 0x00001FFE00,
                'shift': 9
            },
            'counter_a': {
                'mask': 0x00000001FF,
                'shift': 0
            },
            'counter_b': {
                'mask': 0x1FFE00,
                'shift': 9
            },
            'test_pattern': {
                'mask': 0xff,  # should be 9 bits according to manual, but bit 9 is (almost) always 1
                'shift': 0
            },
            # generic portion of the data
            'data': {
                'mask': 0x001FFFFFFF,
                'shift': 0
            },
            'toa': {
                'mask': 0x1ff80000,
                'shift': 19
            },
            'tot': {
                'mask': 0x0007fc00,
                'shift': 10
            },
            'cal': {
                'mask': 0x000003ff,
                'shift': 0
            }
        },
        'trailer': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'chipid': {
                'mask': 0x7FFFC00000,
                'shift': 24
            },
            'status': {
                'mask': 0x00003F0000,
                'shift': 16
            },
            'hits': {
                'mask': 0x000000FF00,
                'shift': 8
            },
            'crc': {
                'mask': 0x00000000FF,
                'shift': 0
            }
        },
        'filler': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'l1counter': {
                'mask': 0x00003FC000,
                'shift': 14
            },
            'ebs': {
                'mask': 0x0000003000,
                'shift': 12
            },
            'bcid': {
                'mask': 0x0000000FFF,
                'shift': 0
            }
        }
    }
}

def create_tamalero_decoder(fmt_dict):
    """Factory function with pre-compiled extractor lambdas."""

    # 1. Pre-calculate ID rules
    id_rules = [
        (id_name, val['frame'], val['mask'])
        for id_name, val in fmt_dict['identifiers'].items()
    ]

    # 2. Build the rule lists
    raw_rules = {}
    for data_type, fields in fmt_dict['data'].items():
        raw_rules[data_type] = [
            (d, fields[d]['mask'], fields[d]['shift']) for d in fields
        ]

    raw_rules['data'] = [
        (d, fmt_dict['data']['data'][d]['mask'], fmt_dict['data']['data'][d]['shift'])
        for d in fmt_dict['types']
    ]

    # 3. PRE-COMPILE EXTRACTORS
    # We turn the rule lists into highly optimized lambda functions.
    # The default argument `r=rules` binds the specific ruleset to the lambda's local scope,
    # meaning the lambda doesn't have to look up `rules` in the outer scope.
    extractors = {
        d_type: lambda v, r=rules: {d: (v & m) >> s for d, m, s in r}
        for d_type, rules in raw_rules.items()
    }

    # 4. Define the read function
    def read(val, quiet=True):
        data_type = None

        # Identify type
        for id_name, frame, mask in id_rules:
            if frame == (val & mask):
                data_type = id_name
                break

        if data_type is None:
            if not quiet:
                print("Found data of type None:", val)
            return None, {}

        # O(1) LOOKUP & EXECUTION: Instantly generate the dictionary
        res = extractors[data_type](val)

        res['raw'] = hex(val & 0xFFFFFFFFFF)
        res['raw_full'] = hex(val)
        res['meta'] = hex((val >> 40) & 0xFFFFFF)

        if not quiet:
            print(f"Found data of type {data_type}:", res)

        return data_type, res

    return read

## --------------------------------------
def merge_words(res):
    """
    Optimized merge_words: Converts to NumPy array only once and prevents
    unnecessary memory copying during bitwise operations. (Optimization 1)
    """
    if not res:
        return []

    arr0 = np.array(res[0::2], dtype=np.uint64)
    arr1 = np.array(res[1::2], dtype=np.uint64)

    len_cut = min(len(arr0), len(arr1))
    arr0, arr1 = arr0[:len_cut], arr1[:len_cut]

    empty_frame_mask = arr0 > 256

    # Return as a standard python list to feed the generator nicely
    return (arr0[empty_frame_mask] | (arr1[empty_frame_mask] << 32)).tolist()

def extract_event_data(unpacked_data_list):
    """
    Single-pass columnar accumulator that builds BOTH hit and status data simultaneously.
    Allows status generation even if the packet contains zero hit data.
    """
    # Pre-allocate dictionaries for both DataFrames
    hit_cols = {
        'evt': [], 'bcid': [], 'l1a_counter': [], 'ea': [],
        'row': [], 'col': [], 'toa': [], 'tot': [], 'cal': [], 'elink': []
    }

    status_cols = {
        'evt': [], 'bcid': [], 'l1a_counter': [], 'elink': [],
        'sof': [], 'eof': [], 'full': [], 'any_full': [], 'global_full': [],
        'chipid': [], 'status': [], 'hits': [], 'crc': []
    }

    pending_packets = {}
    event_counter = -1
    current_bcid = -1
    current_l1acounter = -1

    for record_type, record_data in unpacked_data_list:
        if not record_type or not record_data:
            continue

        elink = record_data.get('elink')
        if elink is None:
            continue

        if record_type == 'header':
            pending_packets[elink] = {'header': record_data, 'data': []}

        elif record_type == 'data':
            if elink in pending_packets:
                pending_packets[elink]['data'].append(record_data)

        elif record_type == 'trailer':
            if elink in pending_packets:
                packet = pending_packets.pop(elink)

                packet_bcid = packet['header'].get('bcid')
                packet_l1acounter = packet['header'].get('l1counter')

                if packet_bcid is None:
                    continue

                # Event counter logic
                if packet_bcid != current_bcid and packet_l1acounter != current_l1acounter:
                    event_counter += 1
                    current_bcid = packet_bcid
                    current_l1acounter = packet_l1acounter
                else:
                    current_bcid = packet_bcid
                    current_l1acounter = packet_l1acounter

                # 1. POPULATE HIT COLUMNS
                # If packet['data'] is empty, this loop safely skips and nothing is appended to hit_cols
                for data_hit in packet['data']:
                    hit_cols['evt'].append(event_counter)
                    hit_cols['bcid'].append(current_bcid)
                    hit_cols['l1a_counter'].append(current_l1acounter)
                    hit_cols['ea'].append(data_hit.get('ea'))
                    hit_cols['row'].append(data_hit.get('row_id'))
                    hit_cols['col'].append(data_hit.get('col_id'))
                    hit_cols['toa'].append(data_hit.get('toa'))
                    hit_cols['tot'].append(data_hit.get('tot'))
                    hit_cols['cal'].append(data_hit.get('cal'))
                    hit_cols['elink'].append(data_hit.get('elink'))

                # 2. POPULATE STATUS COLUMNS
                # This always runs for every valid header-trailer pair, regardless of hit count
                status_cols['evt'].append(event_counter)
                status_cols['bcid'].append(current_bcid)
                status_cols['l1a_counter'].append(current_l1acounter)
                status_cols['elink'].append(packet['header'].get('elink'))
                status_cols['sof'].append(record_data.get('sof'))
                status_cols['eof'].append(record_data.get('eof'))
                status_cols['full'].append(record_data.get('full'))
                status_cols['any_full'].append(record_data.get('any_full'))
                status_cols['global_full'].append(record_data.get('global_full'))
                status_cols['chipid'].append(record_data.get('chipid'))
                status_cols['status'].append(record_data.get('status'))
                status_cols['hits'].append(record_data.get('hits'))
                status_cols['crc'].append(record_data.get('crc'))

    return hit_cols, status_cols

## --------------------------------------
def process_tamalero_outputs(input_files: list):

    start_time = time.monotonic()
    print("\n" + "="*35)
    print("Decode binary format to feather")
    print("="*35)

    all_merged_data = []  # Changed name to reflect it's a flat list, not arrays
    print("\n[1/5] Reading and merging binary data...")

    for ifile in input_files:
        with open(ifile, 'rb') as f:
            bin_data = f.read()
            # Fast binary unpacking
            raw_data = struct.unpack(f'<{len(bin_data)//4}I', bin_data)
            del bin_data

        # Merge data and add directly to our master flat list
        all_merged_data.extend(merge_words(raw_data))
        del raw_data

    print(f"  -> Total words merged: {len(all_merged_data):,}")

    print("\n[2/5] Initializing functional decoder...")
    fast_read_func = create_tamalero_decoder(format_dict)

    print("\n[3/5] Decoding stream and accumulating columns...")
    unpacked_data_gen = (fast_read_func(x) for x in all_merged_data)
    hit_cols, status_cols = extract_event_data(unpacked_data_gen)

    # Clean up the master list to free massive amounts of RAM
    del all_merged_data
    del unpacked_data_gen
    gc.collect()

    print("\n[4/5] Constructing initial DataFrames...")
    hit_df = pd.DataFrame(hit_cols)
    status_df = pd.DataFrame(status_cols)

    del hit_cols
    del status_cols
    gc.collect()

    print("\n[5/5] Downcasting and post-processing...")
    # --- Post-processing for Hit DataFrame ---
    if not hit_df.empty:
        hit_df['board'] = hit_df['elink'] // 4
        hit_df.drop(columns=['elink'], inplace=True)

        dtype_map = {
            'evt': np.uint32, 'bcid': np.uint16, 'l1a_counter': np.uint16,
            'ea': np.uint8, 'board': np.uint8,
            'row': np.uint8, 'col': np.uint8,  # Changed to Unsigned ints (uint8)
            'toa': np.uint16, 'tot': np.uint16, 'cal': np.uint16,
        }
        hit_df = hit_df.astype(dtype_map, errors='ignore')
    else:
        print("No hit data found.")

    # --- Post-processing for Status DataFrame ---
    if not status_df.empty:
        status_df['board'] = status_df['elink'] // 4
        status_df.drop(columns=['elink'], inplace=True)

        status_dtype_map = {
            'evt': np.uint32, 'bcid': np.uint16, 'l1a_counter': np.uint16,
            'sof': np.uint8, 'eof': np.uint8,
            'full': np.uint8, 'any_full': np.uint8, 'global_full': np.uint8,
            'chipid': np.uint16, 'status': np.uint8, 'hits': np.uint8, 'crc': np.uint8,
            'board': np.uint8,
        }
        status_df = status_df.astype(status_dtype_map, errors='ignore')
    else:
        print("No status data found.")

    elapsed_time = time.monotonic() - start_time
    print("\n" + "="*35)
    print(f"Processing complete in {elapsed_time:.2f} seconds.")
    print("="*35 + "\n")

    return hit_df, status_df