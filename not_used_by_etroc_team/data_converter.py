#!/usr/bin/env python3
import struct
import argparse
import numpy as np
import pandas as pd
import awkward as ak
# from tamalero.FIFO import merge_words
from tamalero.DataFrame import DataFrame

def merge_words(res): #TODO: Does this need to be it's own definition here, or can we use the one from tamalero.FIFO
    empty_frame_mask = np.array(res[0::2]) > (2**8)  # masking empty fifo entries
    len_cut = min(len(res[0::2]), len(res[1::2]))  # ensuring equal length of arrays downstream
    if len(res) > 0:
        return list (np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

def event_merger(window_df,merged_idx):
    # Removing events that have been merged already from current window
    window_df = window_df[~window_df.index.isin(merged_idx[merged_idx].index)]
    # Finding events to be merged and group them on their l1counter value
    merge_idx = window_df.duplicated(subset=["l1counter"],keep=False)
    unique_df = window_df[merge_idx].groupby("l1counter",as_index=False).agg(list)
    
    # Clean unique_df (probably a "cleaning" function is needed)
    unique_df["data_type"] = unique_df["data_type"].apply(merge_datatype)
    unique_df["chipid"] = [val[0] for val in unique_df["chipid"]]
    unique_df["status"] = [val[0] for val in unique_df["status"]]
    unique_df['hits'] = [sum(a) for a in unique_df['hits']]
    unique_df['ea'] = unique_df['ea'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['bcid'] = unique_df['bcid'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['col_id'] = unique_df['col_id'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['row_id'] = unique_df['row_id'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['col_id2'] = unique_df['col_id2'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['row_id2'] = unique_df['row_id2'].explode().explode().dropna().groupby(level=0).agg(list)
    unique_df['counter_a'] = unique_df['counter_a'].explode().explode().dropna().groupby(level=0).agg(list)
    
    #prepare the new window of merged events and replace the input one
    new_df = window_df.drop_duplicates('l1counter',keep='first')
    update_df = new_df[["l1counter"]].merge(unique_df,on="l1counter",how="left")
    new_idx = new_df.index
    new_df.set_index("l1counter",inplace=True)
    pd.options.mode.chained_assignment = None # TODO: dirty trick to avoid the SetWithCopy warning: double check everything is fine!
    new_df.update(update_df.set_index("l1counter"))
    pd.options.mode.chained_assignment = 'warn'
    new_df = new_df.reset_index().set_index(new_idx)
    return new_df,merge_idx

def merge_datatype(data_string):
        hd=  ["header"]
        datas = [data for dt in data_string for data in dt if data == "data"]
        tr= ['trailer']
        return hd+datas+tr

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='output/output_example.dat', help="Binary file to read from")
    args = argParser.parse_args() 

    df = DataFrame('ETROC2')
    with open(args.input, 'rb') as f:
        print("Reading from {}".format(args.input))
        bin_data = f.read()
        raw_data = struct.unpack('<{}I'.format(int(len(bin_data)/4)), bin_data)

    merged_data = merge_words(raw_data)
    unpacked_data = [ df.read(x) for x in merged_data ]

    import time
    start = time.process_time()
    #TODO: this is a bit convoluted, at some point it should be cleaned
    # gymnastic to get an awkward array with header,[datas],trailer per each row
    tuple_df = pd.DataFrame.from_dict(unpacked_data)
    data_df = pd.json_normalize(tuple_df[1][:])
    data_df["data_type"] = tuple_df[0][:]
    event_df = data_df.groupby(data_df["elink"].diff().ne(0).cumsum(),as_index=False).agg(list)

    # pandas implementation:
    #------------------------
    print( "time for loading the Dataframe {}".format(round(time.process_time() - start, 2 )))
    Rstart = start
    start=time.process_time()

    #cleaning the df structure  
    for f in event_df.columns: # Here I am merging the fields that are identical across header,hits,trailer 
        maxes = ak.max(event_df[f],axis=-1)
        minmus = ak.min(event_df[f],axis =-1 )
        if ak.all( maxes == minmus ):
            event_df[f] = minmus
    print( "time for cleaning identical columns {} cumulative {}".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
    start=time.process_time()
    # merging unique fields
    event_df["l1counter"] = ak.sum(ak.nan_to_num(event_df["l1counter"], nan=0.0),axis=-1 )
    event_df["status"] = ak.sum(ak.nan_to_num(event_df["status"], nan=0.0),axis=-1 )
    event_df["chipid"] = ak.sum(ak.nan_to_num(event_df["chipid"], nan=0.0),axis=-1 )
    event_df["hits"] = ak.sum(ak.nan_to_num(event_df["hits"], nan=0.0),axis=-1 )
    event_df["type"] = ak.sum(ak.nan_to_num(event_df["type"], nan=0.0),axis=-1 )
    event_df["crc"] = ak.sum(ak.nan_to_num(event_df["crc"], nan=0.0),axis=-1 )
    print( "time for cleaning unique columns {} cumulative {}".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
    start=time.process_time()
    # dropping NaNs 
    event_df['bcid'] = event_df['bcid'].explode().dropna().groupby(level=0).agg(list)
    event_df['ea'] = event_df['ea'].explode().dropna().groupby(level=0).agg(list)
    event_df['counter_a'] = event_df['counter_a'].explode().dropna().groupby(level=0).agg(list)
    event_df['col_id'] = event_df['col_id'].explode().dropna().groupby(level=0).agg(list)
    event_df['row_id'] = event_df['row_id'].explode().dropna().groupby(level=0).agg(list)
    event_df['col_id2'] = event_df['col_id2'].explode().dropna().groupby(level=0).agg(list)
    event_df['row_id2'] = event_df['row_id2'].explode().dropna().groupby(level=0).agg(list)

    print ( "time for removing nans columns {} cumulative {}".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
    start=time.process_time()

    # awkward implementation: 
    #------------------------
    # TODO if you can find an awkward way for groupby it would be interesting to compare the performances  
    # event_ak = ak.Array({name: event_df[name].values for name in event_df.columns})
    # for f in event_ak.fields: # Here I am merging the fields that are identical across header,hits,trailer 
    #     maxes = ak.max(event_ak[f],axis=-1)
    #     minmus = ak.min(event_ak[f],axis =-1 )
    #     if ak.all( maxes == minmus ):
    #         event_ak[f] = minmus
    #         event_df[f] = minmus
    # event_ak["l1counter"] = ak.sum(ak.nan_to_num(event_ak["l1counter"], nan=0.0),axis=-1 )
    # event_ak["status"] = ak.sum(ak.nan_to_num(event_ak["status"], nan=0.0),axis=-1 )
    # event_ak["chipid"] = ak.sum(ak.nan_to_num(event_ak["chipid"], nan=0.0),axis=-1 )
    # event_ak["hits"] = ak.sum(ak.nan_to_num(event_ak["hits"], nan=0.0),axis=-1 )
    # event_ak["type"] = ak.sum(ak.nan_to_num(event_ak["type"], nan=0.0),axis=-1 )
    # event_ak["crc"] = ak.sum(ak.nan_to_num(event_ak["crc"], nan=0.0),axis=-1 )
   
    # merging events in a 100 event window, with 10 events overlap
    window_dfs = []
    window = 100
    overlap = 10
    old_idx = pd.Series([i<-1 for i in range(window)]) #First dummy set of merged indeces
 
    for i in range(0, len(event_df) - window, window - overlap) :
        window_df = event_df[i:i+window]
        new_df,merge_idx = event_merger(window_df,old_idx) 
        window_dfs.append(new_df)
        old_idx = merge_idx
        last = i+window

    last_df,_=event_merger(event_df[last:],old_idx) # merging the ramaining events ()
    window_dfs.append(last_df)
   
    # stitching evnt windows together, dropping overlaps
    merged_df = pd.concat(window_dfs)
    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]

    print( "time for event merging {} cumulative {}".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
    start=time.process_time()
 
    # TODO extend consistency checks
    nHits_check= merged_df["col_id"].str.len().fillna(0) == merged_df["hits"]
    cols_check =merged_df["col_id"].fillna(-99) == merged_df["col_id2"].fillna(-99)
    raws_check = merged_df["row_id"].fillna(-99) == merged_df["row_id2"].fillna(-99)
    header_check = merged_df.data_type.str[0] == "header"
    headerUnique_check =  merged_df.data_type.str[1:] != "header"
    trailer_check =  merged_df.data_type.str[-1] == "trailer"
    trailerUnique_check =  merged_df.data_type.str[:-1] != "trailer"
    data_type_check = header_check & headerUnique_check & trailer_check & trailerUnique_check
    # counter_a_check check that everytime a pixel has a hit, its counter increases by one 
    hits_df = merged_df[merged_df["hits"]>0]
    cols=hits_df['col_id'].apply(lambda x : list(map(int,x)))
    rows=hits_df['row_id'].apply(lambda x : list(map(int,x)))
    counters=hits_df['counter_a'].apply(lambda x : list(map(int,x)))
    counter_ak = ak.Array({"col_id":cols,"row_id":rows,"counter_a":counters})
    for i in range(16):
        for j in range(16):
            print(i,j,"\n")
            pixel_counter=ak.flatten(counter_ak['counter_a'][(counter_ak['col_id']== i) & (counter_ak['row_id']==j)])
            for c in pixel_counter:
                print (c)
            counter_series= pd.Series(pixel_counter.to_numpy())
            print(counter_series.is_monotonic_increasing)
    # counter_a_check = 
    total_check= nHits_check & cols_check & raws_check & data_type_check

    merged_df["valid"] = total_check
    print( "time for consistency checks {} cumulative {}".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
    start=time.process_time()
    merged_df.astype(str).to_parquet("output/converted_data.parquet") # parquet do not likes mixups of list/scalar 
    print( "time for writing the output {} cumulative {} \nAll done".format(round(time.process_time() - start, 2 ), round(time.process_time() - Rstart, 2 )))
