# simple script to compare data coming from different elinks
# written to check how data is being mirrored...

from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.utils import header, get_kcu
from tamalero.FIFO import just_read_daq
from tamalero.DataFrame import DataFrame

from copy import deepcopy
from itertools import combinations

# reorganize stream into a dictionary with key as elink num
# value of dict is a list of parsed header, trailer, data words,
# with only ['elink'] popped.
def repack(stream):
    data_by_elink = {}
    for i in range(len(stream)):
        (datatype, datacontent) = stream[i]
        elink = datacontent['elink']
        # if it's a header, put everything up to the trailer in a list
        # if not a header, it should already be added... skip.
        if (datatype == 'header') and (not elink in data_by_elink.keys()):
            elink_next = elink
            datatype_next = datatype
            datacontent_next = datacontent
            data = []
            j = 0
            while (elink_next == elink) and (i+j < len(stream)):
                # while data word is from same elink...
                data_to_append = deepcopy(datacontent_next)
                data_to_append.pop('elink')
                #if 'crc' in data_to_append:
                #    data_to_append.pop('crc')
                data.append(data_to_append) # add to list
                if datatype_next == 'trailer':
                    break
                else:
                    # move on to next entry in list
                    j += 1
                    (datatype_next, datacontent_next) = stream[i+j]
                    elink_next = datacontent_next['elink']
            # add to the dictionary
            data_by_elink[elink] = data
    return data_by_elink

def print_data(elink_data):
    print('KEY: #   - elink num,   s    - sof,       e - eof,  f - full,   a - any_full,')
    print('     g   - global_full, l1c  - l1counter, t - type, S - status, h - hits,')
    print('     c_a - counter_a,   data - list of all row,cols with hits')
    print('-----------------------------------------------------------------------------')
    print(' # | s e f a g l1c t bcid chipid S  h crc c_a | data')
    print('-----------------------------------------------------------------------------')
    for elink in elink_data:
        words = elink_data[elink]
        h = words[0] #header
        t = words[-1] #trailer
        if len(words) > 2: #dataword
            c_a = words[1]['counter_a']
            data = ''
            for dataword in words[1:-2]:
                if not len(data) == 0:
                    data = data + '/'
                data = data+str(dataword['row_id'])+ ','+str(dataword['col_id'])
        else:
            c_a  = '-'
            data = '-'
        print(f'{elink:>2} | {h["sof"]:>1} {h["eof"]:>1} {h["full"]:>1} {h["any_full"]:>1}'\
              f' {h["global_full"]:>1} {h["l1counter"]:>3} {h["type"]:>1} {h["bcid"]:>4}'\
              f' {t["chipid"]:>6} {t["status"]:>1} {t["hits"]:>2} {t["crc"]:>3}'\
              f' {c_a:3} | {data}')
    return

# group elink numbers with identical data into its own lists.
# TAKES : a dict with the elink # as the key and list of content as value
# RETURNS : 2D list of elink #s grouped into elinks with identical content
def compare(elink_data, print_all):
    result = []
    for elink in elink_data:
        # skip if we already looked at it
        if any(elink in sublist for sublist in result):
            continue
        group = [elink]
        for elink2 in elink_data:
            if (not elink == elink2) and (elink_data[elink] == elink_data[elink2]):
                group.append(elink2)
        result.append(group)
    if print_all:
        print('-----------------------------------------------------------------------------')
        print('Identical elinks : ' + str(result))
    return result

# print statistics of similarity between elinks across all L1As
# TAKES: 3D list, i.e. a list of results of each L1A,
# where the result of each L1A is a 2D list, list of elink nums grouped into one list.
def comparison_stats(data):
    N_l1a = len(data)
    print(f'\n*** Comparision Statistics for {N_l1a} L1As ***')
    print('prints # of instances two elinks have identical data')
    # some initialization...
    elinks = [j for i in data[0] for j in i]
    elinks.sort()
    all_combos = combinations(elinks, 2)
    stat = {}
    for combo in all_combos:
        stat[combo] = 0
    # now, iterate through all L1As and count
    for l1a in data:
        for identicals in l1a:
            combos_to_incr = combinations(identicals, 2)
            for combo in combos_to_incr:
                stat[combo] += 1
    # print all the results
    for combo in stat:
        if stat[combo] == 0:
            continue
        print(f'{str(combo):>8} : {stat[combo]*100/N_l1a:.1f}% ({stat[combo]}/{N_l1a})')
    print('\n=========== >99% ===========')
    for combo in stat:
        if stat[combo] > (N_l1a*0.99):
            print(f'{str(combo):>8} : {stat[combo]*100/N_l1a:.1f}% ({stat[combo]}/{N_l1a})')
    return

if __name__ == '__main__':
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--kcu', action='store', default="192.168.0.10", help="Specify the IP address for KCU")
    argParser.add_argument('--etroc', action='store', default='ETROC2', help='Select ETROC version')
    argParser.add_argument('--triggers', action='store', default=10, help='How many L1As?')
    argParser.add_argument('--compare', action='store_true', default=False, help='Check which links have the same data?')
    argParser.add_argument('--print_all', action='store_true', default=False, help='Print results for all L1As? Not recommended for large numbers of L1A.')
    argParser.add_argument('--control_hub', action='store_true', default=False, help="Use control hub for communication?")
    argParser.add_argument('--host', action='store', default='localhost', help="Specify host for control hub")
    args = argParser.parse_args()

    if not (args.compare or args.print_all):
        raise Exception('Did you want to use either --compare, --print_all, or both? This code wont tell you much without them.')

    kcu = get_kcu(args.kcu, control_hub=args.control_hub, host=args.host)
    rb_0 = ReadoutBoard(0, kcu=kcu)
    df = DataFrame(args.etroc)

    comparison_data = []
    for i in range(int(args.triggers)):
        stream = [df.read(j) for j in just_read_daq(rb_0, 2, 0)]
        elink_data = repack(stream)
        if args.print_all:
            print_data(elink_data)
        if args.compare:
            comparison_data.append(compare(elink_data, args.print_all))
        if args.print_all:
            print('=============================================================================')
    if args.compare:
        comparison_stats(comparison_data)
