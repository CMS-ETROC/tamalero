import pandas as pd
import numpy as np
import struct

from natsort import natsorted
import os
from tqdm import tqdm

## --------------------------------------
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

## --------------------------------------
class TamaleroDF:
    def __init__(self):
        self.format = format_dict

    def get_bytes(self, word, format_order):
        output_bytes = []
        if self.format_order['bitorder'] == 'normal':
            shifts = [32, 24, 16, 8, 0]
        elif self.format_order['bitorder'] == 'reversed':
            shifts = [0, 8, 16, 24, 32]
        for shift in shifts:
            output_bytes.append((word >> shift) & 0xFF)
        if format_order:
            return [ '{0:0{1}x}'.format(b,2) for b in output_bytes ]
        else:
            return output_bytes

    def get_trigger_words(self, format=False):
        return \
            self.get_bytes(self.format['identifiers']['header']['frame'], format=format)  # FIXME check that this still works with FW > v1.2.0

    def get_trigger_masks(self, format=False):
        return \
            self.get_bytes(self.format['identifiers']['header']['mask'], format=format)  # FIXME check that this still works with FW > v1.2.0

    def read(self, val, quiet=True):
        data_type = None
        for id in self.format['identifiers']:
            if self.format['identifiers'][id]['frame'] == (val & self.format['identifiers'][id]['mask']):
                data_type = id
                break

        res = {}
        if data_type == None:
            if not quiet:
                print ("Found data of type None:", val)
            return None, res

        if data_type == 'data':
            datatypelist = self.format['types']
        else:
            datatypelist = self.format['data'][data_type]

        for d in datatypelist:
            res[d] = (val & self.format['data'][data_type][d]['mask']) >> self.format['data'][data_type][d]['shift']

        if data_type == 'header':
            self.type = res['type']
        res['raw'] = hex(val&0xFFFFFFFFFF)
        res['raw_full'] = hex(val)
        res['meta'] = hex((val>>40)&0xFFFFFF)

        if not quiet:
            print (f"Found data of type {data_type}:", res)
        return data_type, res

## --------------------------------------
def merge_words(res):
    empty_frame_mask = np.array(res[0::2]) > (2**8)
    len_cut = min(len(res[0::2]), len(res[1::2]))
    if len(res) > 0:
        return list(np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

## --------------------------------------
def generate_event_rows(unpacked_data_list):
    """
    Generator that processes a list of unpacked data and yields a complete
    row dictionary for each hit. This function does NOT store all the
    output data in memory.
    """
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
            if elink in pending_packets and len(pending_packets[elink]['data']) > 0:
                packet = pending_packets.pop(elink)
                packet_bcid = packet['header'].get('bcid')
                packet_l1acounter = packet['header'].get('l1counter')

                if packet_bcid is None:
                    continue

                # New event if BOTH bcid AND l1acounter don't match
                if packet_bcid != current_bcid and packet_l1acounter != current_l1acounter:
                    event_counter += 1
                    current_bcid = packet_bcid
                    current_l1acounter = packet_l1acounter

                # If either matches, update both to current packet values
                else:
                    current_bcid = packet_bcid
                    current_l1acounter = packet_l1acounter

                for data_hit in packet['data']:
                    # YIELD a dictionary for each individual row
                    yield {
                        'evt': event_counter,
                        'bcid': current_bcid,
                        'l1a_counter': current_l1acounter,
                        'ea': data_hit.get('ea'),
                        'row': data_hit.get('row_id'),
                        'col': data_hit.get('col_id'),
                        'toa': data_hit.get('toa'),
                        'tot': data_hit.get('tot'),
                        'cal': data_hit.get('cal'),
                        'elink': data_hit.get('elink')
                    }
            elif elink in pending_packets:
                del pending_packets[elink]

## --------------------------------------
# UPDATED: The main processing function is now simpler and more robust.
def process_tamalero_outputs(input_files: list):

    all_merged_data = []
    print("Reading and merging data from input files...")
    for ifile in tqdm(input_files):
        with open(ifile, 'rb') as f:
            bin_data = f.read()
            raw_data = struct.unpack(f'<{int(len(bin_data)/4)}I', bin_data)
            del bin_data

        # Merge data and add to a single list for unified processing
        all_merged_data.extend(merge_words(raw_data))
        del raw_data

    print("Decoding data stream...")
    df_decoder = TamaleroDF()
    unpacked_data = (df_decoder.read(x) for x in tqdm(all_merged_data))
    del all_merged_data

    # --- LOW MEMORY STEP 3: Building events with a generator ---
    print("Building events from data...")
    # Create the generator object. This uses almost no memory.
    row_generator = generate_event_rows(unpacked_data)

    # Build the DataFrame directly from the generator. This is memory-efficient
    # as it avoids creating another massive intermediate dictionary.
    print("Building data from generator...")
    final_df = pd.DataFrame.from_records(row_generator)

    # The board mapping and final type casting remain useful
    board_map = {
        0: 0,
        4: 1,
        8: 2,
        12: 3
    }
    if not final_df.empty:
        final_df['board'] = final_df['elink'].map(board_map)
        final_df.drop(columns=['elink'], inplace=True)

        # Set data types for memory efficiency
        dtype_map = {
            'evt': np.uint32,
            'bcid': np.uint16,
            'l1a_counter': np.uint16,
            'ea': np.uint8,
            'board': np.uint8,
            'row': np.int8,
            'col': np.int8,
            'toa': np.uint16,
            'tot': np.uint16,
            'cal': np.uint16,
        }
        final_df = final_df.astype(dtype_map)

    return final_df

def get_last_complete_event(directory, lines_to_read, hits_per_board):
    """
    Loads data and returns the last complete event that satisfies the hits criteria.

    If lines_to_read is -1, it searches through all data from all files.
    Otherwise, it loads the last few lines of the most recent file.

    An event is considered complete if for each board specified in the
    `hits_per_board` dictionary, the number of hits is at least the number required.

    Args:
        directory (str): Path to the directory containing data files.
        lines_to_read (int): Number of 64-bit words to read. Use -1 to read all files.
        hits_per_board (dict): A dictionary specifying the minimum number of hits
                               required per board (e.g., {0: 10, 1: 10}).

    Returns:
        pd.DataFrame: A DataFrame containing a single complete event.
                      Returns an empty DataFrame if no such event is found or an error occurs.
    """
    try:
        files = natsorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        if not files:
            return pd.DataFrame()

        merged_data = []
        if lines_to_read == -1:
            # Full search mode: Read all data from all files.
            print("Full search mode: Reading all files...")
            for file_name in tqdm(files):
                file_path = os.path.join(directory, file_name)
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                if not raw_data:
                    continue
                raw_data_32bit = struct.unpack(f'<{len(raw_data)//4}I', raw_data)
                merged_data.extend(merge_words(raw_data_32bit))
        else:
            # Original mode: Read last few lines of the last file.
            latest_file_path = os.path.join(directory, files[-1])
            word_size = 8  # 8 bytes for a 64-bit unsigned integer
            with open(latest_file_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                if file_size < word_size:
                    return pd.DataFrame()

                num_words_in_file = file_size // word_size
                words_to_read = min(lines_to_read, num_words_in_file)

                if words_to_read == 0:
                    return pd.DataFrame()

                f.seek(-words_to_read * word_size, os.SEEK_END)
                raw_data = f.read()

            if raw_data:
                raw_data_32bit = struct.unpack(f'<{len(raw_data)//4}I', raw_data)
                merged_data = merge_words(raw_data_32bit)

        if not merged_data:
            return pd.DataFrame()

        # Common decoding and searching logic
        df_decoder = TamaleroDF()
        unpacked_data = [df_decoder.read(x) for x in merged_data]

        events = build_events_sequentially(unpacked_data)
        df = pd.DataFrame(events)

        if df.empty:
            return pd.DataFrame()

        # The board mapping from `process_tamalero_outputs`
        board_map = {0: 0, 4: 1, 8: 2, 12: 3}
        df['board'] = df['elink'].map(board_map)

        # An event is identified by a unique evt number from our sequential builder
        event_identifiers = df['evt'].unique()

        # Iterate through events from last to first
        for evt_id in reversed(event_identifiers):
            event_hits = df[df['evt'] == evt_id]
            hit_counts = event_hits['board'].value_counts()

            is_complete = True
            for board, required_hits in hits_per_board.items():
                if hit_counts.get(board, 0) < required_hits:
                    is_complete = False
                    break

            if is_complete:
                # Found a complete event, return it
                return event_hits.drop(columns=['elink'])

    except (FileNotFoundError, IndexError):
        # Handles cases where the directory doesn't exist or is empty
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Return empty dataframe on any other error to prevent crashes
        return pd.DataFrame()

    # If no complete event is found after checking all candidates
    return pd.DataFrame()
