# #!/usr/bin/env python3
# import struct
# import argparse
# import numpy as np
# import pandas as pd
# import awkward as ak
# import json
# import yaml
# from tamalero.DataFrame import DataFrame
# from emoji import emojize
# import os
# import glob

# def merge_words(res):
#     empty_frame_mask = np.array(res[0::2]) > (2**8)  # masking empty fifo entries
#     len_cut = min(len(res[0::2]), len(res[1::2]))  # ensuring equal length of arrays downstream
#     if len(res) > 0:
#         return list (np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
#     else:
#         return []

# def data_dumper(input_file, verbose=False, skip_trigger_check=False, force=False):
#     # NOTE find all files (i.e. layers) for the specified input file
#     df = DataFrame('ETROC2')

#     events_all_rb = []
#     all_runs_good = True
#     missing_l1counter = []

#     in_files = glob.glob(input_file.replace('rb0', 'rb*'))
#     print(in_files)
#     out_files = [x.replace('.dat', '.json') for x in in_files]

#     for irb, f_in in enumerate(in_files):
#         #f_in = f'{here}/ETROC_output/output_run_{args.input}_rb{rb}.dat'
#         with open(f_in, 'rb') as f:
#             print("Reading from {}".format(f_in))
#             bin_data = f.read()
#             raw_data = struct.unpack('<{}I'.format(int(len(bin_data)/4)), bin_data)

#         merged_data = merge_words(raw_data)
#         unpacked_data = map(df.read, merged_data)

#         event       = []
#         counter_h   = []  # double check the number of headers
#         counter_t   = []  # double check the number of trailers
#         l1counter   = []
#         row         = []
#         col         = []
#         tot_code    = []
#         toa_code    = []
#         cal_code    = []
#         elink       = []
#         raw         = []
#         nhits       = []
#         nhits_trail = []
#         chipid      = []
#         crc         = []
#         bcid        = []
#         counter_a   = []

#         header_counter = 0
#         trailer_counter = 0

#         headers = []
#         trailers = []
#         i = 0
#         l1a = -1
#         bcid_t = 9999
#         skip_event = False
#         skip_counter = 0
#         bad_run = False
#         last_missing = False
#         elink_report = {}

#         uuid = []
#         all_raw = []

#         t_tmp = None

#         for t, d in unpacked_data:
#             if d['elink'] not in elink_report:
#                 elink_report[d['elink']] = {'nheader':0, 'nhits':0, 'ntrailer':0}
#             sus = False
#             if d['raw_full'] in all_raw[-50:] and not t in ['trailer', 'filler']:  # trailers often look the same
#                 #print("Potential double counting", t, d)
#                 #all_raw.append(d['raw_full'])
#                 continue
#             if t not in ['trailer', 'filler']:
#                 all_raw.append(d['raw_full'])

#             if t == 'header':
#                 elink_report[d['elink']]['nheader'] += 1
#                 hit_counter = 0
#                 uuid_tmp = d['l1counter'] | d['bcid']<<8
#                 headers.append(d['raw'])
#                 header_counter += 1
#                 #if (abs(d['bcid']-bcid_t)<50) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
#                 #    skip_event = True
#                 #    print("Skipping event")
#                 #    continue
#                 if d['bcid'] != bcid_t and last_missing:
#                     missing_l1counter[-1].append(d['bcid'])
#                     last_missing = False

#                 if d['l1counter'] == l1a:
#                     # this just skips additional headers for the same event
#                     counter_h[-1] += 1
#                     raw[-1].append(d['raw_full'])
#                     if skip_event:
#                         print("Skipping event (same l1a counter)", d['l1counter'], d['bcid'], bcid_t)
#                         continue
#                     pass
#                 #elif d['l1counter'] < l1a and d['l1counter']!=0:
#                 # # NOTE this part is experimental, and removes duplicate events
#                 # # However, we should instead re-implement full consistency checks of header - data - trailer style
#                 #    skip_event = True
#                 #    print("Skipping event", d['l1counter'], d['bcid'], bcid_t)

#                 else:
#                     if abs(l1a - d['l1counter']) not in [1,255] and l1a>=0:
#                         missing_l1counter.append([d['l1counter'], d['bcid'], i, d['l1counter'] - l1a])  # this checks if we miss any event according to the counter
#                         last_missing = True
#                     if uuid_tmp in uuid and abs(i - np.where(np.array(uuid) == uuid_tmp)[0][-1]) < 150:
#                         print("Skipping duplicate event")
#                         skip_counter += 1
#                         skip_event = True
#                         continue
#                     else:
#                         uuid.append(d['l1counter'] | d['bcid']<<8)
#                     #hit_counter = 0
#                     #if (abs(d['bcid']-bcid_t)<40) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
#                     #if (abs(d['bcid']-bcid_t)<500) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
#                     if (((abs(d['bcid']-bcid_t)<150) or (abs(d['bcid']+3564-bcid_t)<50)) and not (d['bcid'] == bcid_t) and not skip_trigger_check):
#                         skip_event = True
#                         print("Skipping event", d['l1counter'], d['bcid'], bcid_t)
#                         skip_counter += 1
#                         continue
#                     else:
#                         skip_event = False
#                     bcid_t = d['bcid']
#                     if (abs(l1a - d['l1counter'])>1) and abs(l1a - d['l1counter'])!=255 and verbose:
#                         print("SUS")
#                         sus = True
#                     l1a = d['l1counter']
#                     event.append(i)
#                     counter_h.append(1)
#                     counter_t.append(0)
#                     l1counter.append(d['l1counter'])
#                     row.append([])
#                     col.append([])
#                     tot_code.append([])
#                     toa_code.append([])
#                     cal_code.append([])
#                     elink.append([])
#                     raw.append([d['raw_full']])
#                     nhits.append(0)
#                     nhits_trail.append([])
#                     chipid.append([])
#                     crc.append([])
#                     bcid.append([d['bcid']])
#                     counter_a.append([])
#                     i += 1
#                     if verbose or sus:
#                         print("New event:", l1a, i, d['bcid'])

#             if t == 'data':
#                 elink_report[d['elink']]['nhits'] += 1
#             if t == 'trailer':
#                 elink_report[d['elink']]['ntrailer'] += 1

#             if t == 'data' and not skip_event:
#                 hit_counter += 1
#                 if 'tot' in d:
#                     tot_code[-1].append(d['tot'])
#                     toa_code[-1].append(d['toa'])
#                     cal_code[-1].append(d['cal'])
#                 elif 'counter_a' in d:
#                     bcid[-1].append(d['bcid'])
#                     counter_a[-1].append(d['counter_a'])
#                 elif 'counter_b' in d:
#                     pass
#                 row[-1].append(d['row_id'])
#                 col[-1].append(d['col_id'])
#                 elink[-1].append(d['elink'])
#                 raw[-1].append(d['raw_full'])
#                 nhits[-1] += 1
#                 if nhits[-1] > 256:
#                     print("This event already has more than 256 hits. Skipping event.")
#                     skip_event = True
#                     continue
#                     #bad_run = True
#                     #break

#             if t == 'trailer' and t_tmp != 'trailer':
#                 trailers.append(d['raw_full'])
#                 trailer_counter += 1
#                 if not skip_event:
#                     try:
#                         counter_t[-1] += 1
#                         if hit_counter > 0:
#                             #print(hit_counter)
#                             #chipid[-1].append(hit_counter*d['chipid'])
#                             chipid[-1] += hit_counter*[d['chipid']]
#                             #print(l1counter[-1], bcid[-1], chipid[-1])
#                         #else:
#                         #    chipid[-1].append()
#                         nhits_trail[-1].append(d['hits'])
#                         raw[-1].append(d['raw'])
#                         crc[-1].append(d['crc'])
#                     except IndexError:
#                         print("Data stream started with a trailer, that is weird.")

#             t_tmp = t

#         if (not bad_run or force) and event:
#             print("Zipping")
#             events = ak.Array({
#                 'event': event,
#                 'l1counter': l1counter,
#                 'nheaders': counter_h,
#                 'ntrailers': counter_t,
#                 'row': row,
#                 'col': col,
#                 'tot_code': tot_code,
#                 'toa_code': toa_code,
#                 'cal_code': cal_code,
#                 'elink': elink,
#                 'raw': raw,
#                 'crc': crc,
#                 'chipid': chipid,
#                 'bcid': bcid,
#                 'counter_a': counter_a,
#                 'nhits': ak.singletons(nhits),
#                 'nhits_trail': ak.sum(ak.Array(nhits_trail), axis=-1),
#             })

#             total_events = len(events)
#             # NOTE the check below is only valid for single ETROC
#             #consistent_events = len(events[((events.nheaders==2)&(events.ntrailers==2)&(events.nhits==events.nhits_trail))])
#             #print(total_events, consistent_events)

#             print(f"Done with {len(events)} events. " + emojize(":check_mark_button:"))
#             #print(f" - skipped {skip_counter/events.nheaders[0]} events that were identified as double-triggered " + emojize(":check_mark_button:"))
#             if header_counter == trailer_counter:
#                 print(f" - found same number of headers and trailers!: {header_counter} " + emojize(":check_mark_button:"))
#             else:
#                 print(f" - found {header_counter} headers and {trailer_counter} trailers. Please check. " + emojize(":warning:"))

#             print(f" - found {len(missing_l1counter)} missing events (irregular increase of L1counter).")
#             if len(missing_l1counter)>0:
#                 print("   L1counter, BCID, event number and step size of these events are:")
#                 if len(missing_l1counter[-1]) == 4:
#                     missing_l1counter[-1].append(-9999) #append dummy value
#                 for ml1,mbcid,mev,mdelta,mbcidt in missing_l1counter:
#                     if mbcidt - mbcid<7:
#                         print("Expected issue because of missing L1A dead time:", ml1, mbcid, mev,mdelta,mbcidt)
#                     else:
#                         print(ml1, mbcid, mev,mdelta,mbcidt)

#             print(f" - Total expected events is {total_events+len(missing_l1counter)}")
#             print(f" - elink report:")
#             print(pd.DataFrame(elink_report))

#             with open(out_files[irb], "w") as f:
#                 json.dump(ak.to_json(events), f)
#             #with open(f"ETROC_output/{args.input}_rb{rb}.json", "w") as f:
#             #    json.dump(ak.to_json(events), f)
#             events_all_rb.append(events)

#             # make some plots
#             import matplotlib.pyplot as plt
#             import mplhep as hep
#             plt.style.use(hep.style.CMS)

#             hits = np.zeros([16, 16])
#             #if rb=='2':
#             #    mask = [(11,5)]
#             mask = []
#             #if rb=='0':
#             #    mask = [(2,4), (3,4), (4,6), (3,11), (6,12)]
#             #if rb=='1':
#             #    mask = [(4,0)]
#             for ev in events:
#                 for row, col in zip(ev.row, ev.col):
#                     if (row, col) not in mask:
#                         hits[row][col] += 1

#             fig, ax = plt.subplots(1,1,figsize=(7,7))
#             cax = ax.matshow(hits)
#             ax.set_ylabel(r'$Row$')
#             ax.set_xlabel(r'$Column$')
#             fig.colorbar(cax,ax=ax)
#             plot_file = f_in.replace('.dat', '_heatmap.png')
#             fig.savefig(plot_file)
#             #fig.savefig(f"ETROC_output/{args.input}_rb{rb}_heatmap.png")

#             # FIXME this only works for a single ETROC right now
#             #if args.dump_mask:
#             #    with open(f"{here}/ETROC_output/mask_run{args.input}.yaml", 'w') as f:
#             #        yaml.dump(hits.tolist(), f)
#             #        #yaml.dump(hits, f)

#             total_hits = np.sum(hits)
#             print("Total number of hits:", total_hits)
#         else:
#             print("Bad run detected. Not creating a json file.")
#             all_runs_good = False
#             #if os.path.isfile(f"{here}/ETROC_output/output_run_{args.input}_rb{rb}.json"):
#             #    os.remove(f"{here}/ETROC_output/output_run_{args.input}_rb{rb}.json")

#     if len(events_all_rb)>1: #or True:
#         event_number = []
#         bcid = []
#         nhits = []
#         row = []
#         col = []
#         chipid = []

#         #sel = ak.flatten(events_all_rb[0].nhits>0)
#         #sel = ak.flatten(ak.ones_like(events_all_rb[0].nhits))
#         sel = ak.flatten(ak.ones_like(events_all_rb[0].nhits, dtype=bool))
#         events_with_hits = len(events_all_rb[0][sel])

#         for rb, events in enumerate(events_all_rb):
#             if rb == 0:
#                 #sel = ak.flatten(events.nhits)>0
#                 event_number = ak.to_list(events[sel].event)
#                 l1counter    = ak.to_list(events[sel].l1counter)
#                 bcid         = ak.to_list(events[sel].bcid)
#                 nhits        = ak.to_list(events[sel].nhits)
#                 row          = ak.to_list(events[sel].row)
#                 col          = ak.to_list(events[sel].col)
#                 toa          = ak.to_list(events[sel].toa_code)
#                 tot          = ak.to_list(events[sel].tot_code)
#                 cal          = ak.to_list(events[sel].cal_code)
#                 elink        = ak.to_list(events[sel].elink)
#                 chipid       = ak.to_list(events[sel].chipid)
#             else:
#                 # loop through rb0 events, and find corresponding entries in the other layers
#                 print(f"Merging events from RB {rb}")
#                 from tqdm import tqdm
#                 with tqdm(total=events_with_hits, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
#                     for i, ev in enumerate(event_number):
#                         # print(i, ev)
#                         for j, tmp_evt in enumerate(events_all_rb[rb][ak.flatten(events_all_rb[rb].bcid + 1 == events_all_rb[0].bcid[ev])]):
#                             # print(j,tmp_evt)
#                         #for j in events_all_rb[rb].event:
#                             if abs(tmp_evt.event - ev)<100:
#                             #if events_all_rb[rb].l1counter[j] == events_all_rb[0].l1counter[i] and events_all_rb[rb].bcid[j] == events_all_rb[0].bcid[i] and abs(j-i) < 100:
#                             #if events_all_rb[rb].bcid[j] == events_all_rb[0].bcid[ev] and abs(j-ev) < 100:
#                                 nhits[i] += ak.to_list(tmp_evt.nhits)
#                                 row[i] += ak.to_list(tmp_evt.row)
#                                 col[i] += ak.to_list(tmp_evt.col)
#                                 tot[i] += ak.to_list(tmp_evt.tot_code)
#                                 toa[i] += ak.to_list(tmp_evt.toa_code)
#                                 cal[i] += ak.to_list(tmp_evt.cal_code)
#                                 elink[i] += ak.to_list(tmp_evt.elink)
#                                 chipid[i] += ak.to_list(tmp_evt.chipid)
#                                 #print("--------------------------")
#                                 #print(f"Found matching event for event {i} in rb {rb} stream")
#                                 break
#                         pbar.update()

#                 #for j in events.event:
#                 #    if events.l1counter[j] == events_all_rb[0].l1counter[j] and events.bcid[j] == events_all_rb[0].bcid[j]:
#                 #        nhits[j].append(events.nhits)
#                 #        row[j].append(events.row)
#                 #        col[j].append(events.col)
#                 #        chipid[j].append(events.chipid)

#         print("Zipping again")
#         events = ak.Array({
#             'event': event_number,
#             'l1counter': l1counter,
#             #'nheaders': counter_h,
#             #'ntrailers': counter_t,
#             'row': row,
#             'col': col,
#             'tot_code': tot,
#             'toa_code': toa,
#             'cal_code': cal,
#             'elink': elink,
#             #'raw': raw,
#             #'crc': crc,
#             'chipid': chipid,
#             'bcid': bcid,
#             #'counter_a': counter_a,
#             'nhits': nhits,
#             #'nhits_trail': ak.sum(ak.Array(nhits_trail), axis=-1),
#         })
#         with open(out_files[0].replace('rb0', 'merged'), "w") as f:
#             json.dump(ak.to_json(events), f)
#         # make a copy that is called rb0 for the merger
#         with open(out_files[0], "w") as f:
#             json.dump(ak.to_json(events), f)
#         print("Done.")

#     if not bad_run and in_files and events_all_rb:
#         return len(events), events
#     else:
#         print(f"NO EVENTS for {in_files}, NOT MAKING JSON AND HITMAPS...")
#         return 0, []

# if __name__ == '__main__':
#     argParser = argparse.ArgumentParser(description = "Argument parser")
#     argParser.add_argument('--input_file', action='store', default='output_test2', help="Binary file to read from")
#     argParser.add_argument('--rbs', action='store', default='0', help="RB numbers")
#     argParser.add_argument('--skip_trigger_check', action='store_true', help="Skip the double trigger check.")
#     argParser.add_argument('--dump_mask', action='store_true', help="Skip the double trigger check.")
#     argParser.add_argument('--verbose', action='store_true', help="Print every event number.")
#     argParser.add_argument('--force', action='store_true', help="Don't care about inconsistencies, force produce output.")
#     args = argParser.parse_args()

#     rbs = args.rbs.split(',')

#     nevents, events = data_dumper(args.input_file, verbose=args.verbose, skip_trigger_check=args.skip_trigger_check, force=args.force)




#!/usr/bin/env python3
import struct
import argparse
import numpy as np
import pandas as pd
import awkward as ak
import json
import yaml
from yaml import Dumper, Loader
from tamalero.DataFrame import DataFrame
from emoji import emojize
import os
import glob

def merge_words(res):
    empty_frame_mask = np.array(res[0::2]) > (2**8)  # masking empty fifo entries
    len_cut = min(len(res[0::2]), len(res[1::2]))  # ensuring equal length of arrays downstream
    if len(res) > 0:
        return list (np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

def data_dumper(
        input_file,
        #output_file,
        verbose=False,
        skip_trigger_check=False,
        force=False,
):
    # NOTE find all files (i.e. layers) for the specified input file
    df = DataFrame('ETROC2')

    events_all_rb = []
    all_runs_good = True
    missing_l1counter = []

    in_files = glob.glob(input_file.replace('rb0', 'rb*'))
    print(in_files)
    out_files = [x.replace('.dat', '.json') for x in in_files]

    for irb, f_in in enumerate(in_files):
        #f_in = f'{here}/ETROC_output/output_run_{args.input}_rb{rb}.dat'
        with open(f_in, 'rb') as f:
            print("Reading from {}".format(f_in))
            bin_data = f.read()
            raw_data = struct.unpack('<{}I'.format(int(len(bin_data)/4)), bin_data)

        merged_data = merge_words(raw_data)
        unpacked_data = map(df.read, merged_data)

        event       = []
        counter_h   = []  # double check the number of headers
        counter_t   = []  # double check the number of trailers
        l1counter   = []
        row         = []
        col         = []
        tot_code    = []
        toa_code    = []
        cal_code    = []
        elink       = []
        raw         = []
        nhits       = []
        nhits_trail = []
        chipid      = []
        crc         = []
        bcid        = []
        counter_a   = []

        header_counter = 0
        trailer_counter = 0

        headers = []
        trailers = []
        i = 0
        l1a = -1
        bcid_t = 9999
        skip_event = False
        skip_counter = 0
        bad_run = False
        last_missing = False
        elink_report = {}

        uuid = []
        all_raw = []

        t_tmp = None

        for t, d in unpacked_data:
            if d['elink'] not in elink_report:
                elink_report[d['elink']] = {'nheader':0, 'nhits':0, 'ntrailer':0}
            sus = False
            if d['raw_full'] in all_raw[-50:] and not t in ['trailer', 'filler']:  # trailers often look the same
                #print("Potential double counting", t, d)
                #all_raw.append(d['raw_full'])
                continue
            if t not in ['trailer', 'filler']:
                all_raw.append(d['raw_full'])

            if t == 'header':
                elink_report[d['elink']]['nheader'] += 1
                hit_counter = 0
                uuid_tmp = d['l1counter'] | d['bcid']<<8
                headers.append(d['raw'])
                header_counter += 1
                #if (abs(d['bcid']-bcid_t)<50) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
                #    skip_event = True
                #    print("Skipping event")
                #    continue
                if d['bcid'] != bcid_t and last_missing:
                    missing_l1counter[-1].append(d['bcid'])
                    last_missing = False

                if d['l1counter'] == l1a:
                    # this just skips additional headers for the same event
                    counter_h[-1] += 1
                    raw[-1].append(d['raw_full'])
                    if skip_event:
                        print("Skipping event (same l1a counter)", d['l1counter'], d['bcid'], bcid_t)
                        continue
                    pass
                #elif d['l1counter'] < l1a and d['l1counter']!=0:
                # # NOTE this part is experimental, and removes duplicate events
                # # However, we should instead re-implement full consistency checks of header - data - trailer style
                #    skip_event = True
                #    print("Skipping event", d['l1counter'], d['bcid'], bcid_t)

                else:
                    if abs(l1a - d['l1counter']) not in [1,255] and l1a>=0:
                        missing_l1counter.append([d['l1counter'], d['bcid'], i, d['l1counter'] - l1a])  # this checks if we miss any event according to the counter
                        last_missing = True
                    if uuid_tmp in uuid and abs(i - np.where(np.array(uuid) == uuid_tmp)[0][-1]) < 150:
                        print("Skipping duplicate event")
                        skip_counter += 1
                        skip_event = True
                        continue
                    else:
                        uuid.append(d['l1counter'] | d['bcid']<<8)
                    #hit_counter = 0
                    #if (abs(d['bcid']-bcid_t)<40) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
                    #if (abs(d['bcid']-bcid_t)<500) and (d['bcid'] - bcid_t)>0 and not (d['bcid'] == bcid_t):
                    if (((abs(d['bcid']-bcid_t)<150) or (abs(d['bcid']+3564-bcid_t)<50)) and not (d['bcid'] == bcid_t) and not skip_trigger_check):
                        skip_event = True
                        print("Skipping event", d['l1counter'], d['bcid'], bcid_t)
                        skip_counter += 1
                        continue
                    else:
                        skip_event = False
                    bcid_t = d['bcid']
                    if (abs(l1a - d['l1counter'])>1) and abs(l1a - d['l1counter'])!=255 and verbose:
                        print("SUS")
                        sus = True
                    l1a = d['l1counter']
                    event.append(i)
                    counter_h.append(1)
                    counter_t.append(0)
                    l1counter.append(d['l1counter'])
                    row.append([])
                    col.append([])
                    tot_code.append([])
                    toa_code.append([])
                    cal_code.append([])
                    elink.append([])
                    raw.append([d['raw_full']])
                    nhits.append(0)
                    nhits_trail.append([])
                    chipid.append([])
                    crc.append([])
                    bcid.append([d['bcid']])
                    counter_a.append([])
                    i += 1
                    if verbose or sus:
                        print("New event:", l1a, i, d['bcid'])

            if t == 'data':
                elink_report[d['elink']]['nhits'] += 1
            if t == 'trailer':
                elink_report[d['elink']]['ntrailer'] += 1

            if t == 'data' and not skip_event:
                hit_counter += 1
                if 'tot' in d:
                    tot_code[-1].append(d['tot'])
                    toa_code[-1].append(d['toa'])
                    cal_code[-1].append(d['cal'])
                elif 'counter_a' in d:
                    bcid[-1].append(d['bcid'])
                    counter_a[-1].append(d['counter_a'])
                elif 'counter_b' in d:
                    pass
                row[-1].append(d['row_id'])
                col[-1].append(d['col_id'])
                elink[-1].append(d['elink'])
                raw[-1].append(d['raw_full'])
                nhits[-1] += 1
                if nhits[-1] > 256:
                    print("This event already has more than 256 hits. Skipping event.")
                    skip_event = True
                    continue
                    #bad_run = True
                    #break

            if t == 'trailer' and t_tmp != 'trailer':
                trailers.append(d['raw_full'])
                trailer_counter += 1
                if not skip_event:
                    try:
                        counter_t[-1] += 1
                        if hit_counter > 0:
                            #print(hit_counter)
                            #chipid[-1].append(hit_counter*d['chipid'])
                            chipid[-1] += hit_counter*[d['chipid']]
                            #print(l1counter[-1], bcid[-1], chipid[-1])
                        #else:
                        #    chipid[-1].append()
                        nhits_trail[-1].append(d['hits'])
                        raw[-1].append(d['raw'])
                        crc[-1].append(d['crc'])
                    except IndexError:
                        print("Data stream started with a trailer, that is weird.")

            t_tmp = t

        if (not bad_run or force) and event:
            print("Zipping")
            events = ak.Array({
                'event': event,
                'l1counter': l1counter,
                'nheaders': counter_h,
                'ntrailers': counter_t,
                'row': row,
                'col': col,
                'tot_code': tot_code,
                'toa_code': toa_code,
                'cal_code': cal_code,
                'elink': elink,
                'raw': raw,
                'crc': crc,
                'chipid': chipid,
                'bcid': bcid,
                'counter_a': counter_a,
                'nhits': ak.singletons(nhits),
                'nhits_trail': ak.sum(ak.Array(nhits_trail), axis=-1),
            })

            total_events = len(events)
            # NOTE the check below is only valid for single ETROC
            #consistent_events = len(events[((events.nheaders==2)&(events.ntrailers==2)&(events.nhits==events.nhits_trail))])
            #print(total_events, consistent_events)

            print(f"Done with {len(events)} events. " + emojize(":check_mark_button:"))
            #print(f" - skipped {skip_counter/events.nheaders[0]} events that were identified as double-triggered " + emojize(":check_mark_button:"))
            if header_counter == trailer_counter:
                print(f" - found same number of headers and trailers!: {header_counter} " + emojize(":check_mark_button:"))
            else:
                print(f" - found {header_counter} headers and {trailer_counter} trailers. Please check. " + emojize(":warning:"))

            print(f" - found {len(missing_l1counter)} missing events (irregular increase of L1counter).")
            if len(missing_l1counter)>0:
                print("   L1counter, BCID, event number and step size of these events are:")
                for ml1,mbcid,mev,mdelta,mbcidt in missing_l1counter:
                    if mbcidt - mbcid<7:
                        print("Expected issue because of missing L1A dead time:", ml1, mbcid, mev,mdelta,mbcidt)
                    else:
                        print(ml1, mbcid, mev,mdelta,mbcidt)

            print(f" - Total expected events is {total_events+len(missing_l1counter)}")
            print(f" - elink report:")
            print(pd.DataFrame(elink_report))

            with open(out_files[irb], "w") as f:
                json.dump(ak.to_json(events), f)
            #with open(f"ETROC_output/{args.input}_rb{rb}.json", "w") as f:
            #    json.dump(ak.to_json(events), f)
            events_all_rb.append(events)

            # make some plots
            import matplotlib.pyplot as plt
            import mplhep as hep
            plt.style.use(hep.style.CMS)

            hits = np.zeros([16, 16])
            #if rb=='2':
            #    mask = [(11,5)]
            mask = []
            #if rb=='0':
            #    mask = [(2,4), (3,4), (4,6), (3,11), (6,12)]
            #if rb=='1':
            #    mask = [(4,0)]
            for ev in events:
                for row, col in zip(ev.row, ev.col):
                    if (row, col) not in mask:
                        hits[row][col] += 1

            fig, ax = plt.subplots(1,1,figsize=(7,7))
            cax = ax.matshow(hits)
            ax.set_ylabel(r'$Row$')
            ax.set_xlabel(r'$Column$')
            fig.colorbar(cax,ax=ax)
            plot_file = f_in.replace('.dat', '_heatmap.png')
            fig.savefig(plot_file)
            #fig.savefig(f"ETROC_output/{args.input}_rb{rb}_heatmap.png")

            # FIXME this only works for a single ETROC right now
            #if args.dump_mask:
            #    with open(f"{here}/ETROC_output/mask_run{args.input}.yaml", 'w') as f:
            #        yaml.dump(hits.tolist(), f)
            #        #yaml.dump(hits, f)

            total_hits = np.sum(hits)
            print("Total number of hits:", total_hits)
        else:
            print("Bad run detected. Not creating a json file.")
            all_runs_good = False
            #if os.path.isfile(f"{here}/ETROC_output/output_run_{args.input}_rb{rb}.json"):
            #    os.remove(f"{here}/ETROC_output/output_run_{args.input}_rb{rb}.json")

    if len(events_all_rb)>1: #or True:
        event_number = []
        bcid = []
        nhits = []
        row = []
        col = []
        chipid = []

        #sel = ak.flatten(events_all_rb[0].nhits>0)
        #sel = ak.flatten(ak.ones_like(events_all_rb[0].nhits))
        sel = ak.flatten(ak.ones_like(events_all_rb[0].nhits, dtype=bool))
        events_with_hits = len(events_all_rb[0][sel])

        for rb, events in enumerate(events_all_rb):
            if rb == 0:
                #sel = ak.flatten(events.nhits)>0
                event_number = ak.to_list(events[sel].event)
                l1counter = ak.to_list(events[sel].l1counter)
                bcid = ak.to_list(events[sel].bcid)
                nhits = ak.to_list(events[sel].nhits)
                row = ak.to_list(events[sel].row)
                col = ak.to_list(events[sel].col)
                toa = ak.to_list(events[sel].toa_code)
                tot = ak.to_list(events[sel].tot_code)
                cal = ak.to_list(events[sel].cal_code)
                elink = ak.to_list(events[sel].elink)
                chipid = ak.to_list(events[sel].chipid)
            else:
                # loop through rb0 events, and find corresponding entries in the other layers
                print(f"Merging events from RB {rb}")
                from tqdm import tqdm
                with tqdm(total=events_with_hits, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                    for i, ev in enumerate(event_number):
                        # print(i, ev)
                        for j, tmp_evt in enumerate(events_all_rb[rb][ak.flatten(events_all_rb[rb].bcid + 1 == events_all_rb[0].bcid[ev])]):
                            # print(j,tmp_evt)
                        #for j in events_all_rb[rb].event:
                            if abs(tmp_evt.event - ev)<100:
                            #if events_all_rb[rb].l1counter[j] == events_all_rb[0].l1counter[i] and events_all_rb[rb].bcid[j] == events_all_rb[0].bcid[i] and abs(j-i) < 100:
                            #if events_all_rb[rb].bcid[j] == events_all_rb[0].bcid[ev] and abs(j-ev) < 100:
                                nhits[i] += ak.to_list(tmp_evt.nhits)
                                row[i] += ak.to_list(tmp_evt.row)
                                col[i] += ak.to_list(tmp_evt.col)
                                tot[i] += ak.to_list(tmp_evt.tot_code)
                                toa[i] += ak.to_list(tmp_evt.toa_code)
                                cal[i] += ak.to_list(tmp_evt.cal_code)
                                elink[i] += ak.to_list(tmp_evt.elink)
                                chipid[i] += ak.to_list(tmp_evt.chipid)
                                #print("--------------------------")
                                #print(f"Found matching event for event {i} in rb {rb} stream")
                                break
                        pbar.update()

                #for j in events.event:
                #    if events.l1counter[j] == events_all_rb[0].l1counter[j] and events.bcid[j] == events_all_rb[0].bcid[j]:
                #        nhits[j].append(events.nhits)
                #        row[j].append(events.row)
                #        col[j].append(events.col)
                #        chipid[j].append(events.chipid)

        print("Zipping again")
        events = ak.Array({
            'event': event_number,
            'l1counter': l1counter,
            #'nheaders': counter_h,
            #'ntrailers': counter_t,
            'row': row,
            'col': col,
            'tot_code': tot,
            'toa_code': toa,
            'cal_code': cal,
            'elink': elink,
            #'raw': raw,
            #'crc': crc,
            'chipid': chipid,
            'bcid': bcid,
            #'counter_a': counter_a,
            'nhits': nhits,
            #'nhits_trail': ak.sum(ak.Array(nhits_trail), axis=-1),
        })
        with open(out_files[0].replace('rb0', 'merged'), "w") as f:
            json.dump(ak.to_json(events), f)
        # make a copy that is called rb0 for the merger
        with open(out_files[0], "w") as f:
            json.dump(ak.to_json(events), f)
        print("Done.")

    if not bad_run and in_files and events_all_rb:
        return len(events), events
    else:
        print(f"NO EVENTS for {in_files}, NOT MAKING JSON AND HITMAPS...")
        return 0, []

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input_file', action='store', default='output_test2', help="Binary file to read from")
    argParser.add_argument('--rbs', action='store', default='0', help="RB numbers")
    argParser.add_argument('--skip_trigger_check', action='store_true', help="Skip the double trigger check.")
    argParser.add_argument('--dump_mask', action='store_true', help="Skip the double trigger check.")
    argParser.add_argument('--verbose', action='store_true', help="Print every event number.")
    argParser.add_argument('--force', action='store_true', help="Don't care about inconsistencies, force produce output.")
    args = argParser.parse_args()

    rbs = args.rbs.split(',')

    nevents, events = data_dumper(
        args.input_file,
        #output_file,
        verbose=args.verbose,
        skip_trigger_check=args.skip_trigger_check,
        force=args.force,
    )