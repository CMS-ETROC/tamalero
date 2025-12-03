import argparse
import struct
import numpy as np
import awkward as ak
from tamalero.DataFrame import DataFrame

# DISCLAIMER
# This is still work in progress, when finalized it should be included in the 
# data_converter.py script

def merge_words(res):
    empty_frame_mask = np.array(res[0::2]) > (2**8)  # masking empty fifo entries
    len_cut = min(len(res[0::2]), len(res[1::2]))  # ensuring equal length of arrays downstream
    if len(res) > 0:
        return list (np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

#Define XOR for same length bits 
def xor(a, b):
    # initialize result
    result = []
    # Traverse all bits, if bits are same, then XOR is 0, else 1
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
 
    return ''.join(result)

# Performs Modulo-2 division
def mod2div(dividend, divisor):
    
    pick = len(divisor) # Number of bits to be XORed at a time.
    tmp = dividend[0: pick] # Slicing the dividend to appropriate length 

    while pick < len(dividend):
        if tmp[0] == '1':
            # replace the dividend by the result of XOR and pull 1 bit down
            tmp = xor(divisor, tmp) + dividend[pick]
        else:   
            # If the leftmost bit of the dividend (or the part used in 
            # each step) is 0, the step cannot use the regular divisor; 
            # we need to use an all-0s divisor.
            tmp = xor('0'*pick, tmp) + dividend[pick]
        # increment pick to move further
        pick += 1
    # Last n bits...
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)
 
    crc = tmp
    return crc

if __name__ == '__main__':

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='output/output_example.dat', help="Binary file to read from")
    args = argParser.parse_args() 

    df = DataFrame('ETROC2')

    with open(args.input, 'rb') as f:
        bin_data = f.read()
        raw_data = struct.unpack('<{}I'.format(int(len(bin_data)/4)), bin_data)

    merged_data = merge_words(raw_data)
    #handy functions
    etrocmask = np.vectorize(lambda x : x & 0xFFFFFFFFFF) #mask 64 bits words to 40bits
    hexstr5=np.vectorize(lambda x: f'{x:05x}') #40bits 5 hex
    binstr40=np.vectorize(lambda x: f'{x:040b}') #40bits
    words = ak.Array(hexstr5(etrocmask(merged_data)))
    bitwords = ak.Array(binstr40(etrocmask(merged_data)))
    #FIXME vectorize this abomination
    header_idx=[]
    for i,word in enumerate(words):
        if '3c5c' in word: header_idx.append(i)
    header_idx.append(len(words))
    dataframes = ak.unflatten(words,np.diff(header_idx))
    bitframes = ak.unflatten(bitwords,np.diff(header_idx))
    
    merged_frames = ["".join(x) for x in bitframes]
    poly = '100101111'
    crc_checks = np.array([mod2div(frame,poly) for frame in merged_frames])
