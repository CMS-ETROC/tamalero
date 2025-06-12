# ╔═╗╔═╗╔═╗╔╦╗╦ ╦╔═╗╦═╗╔═╗  ╔═╗╔╦╗╦═╗╔═╗╔═╗╔═╗
# ╚═╗║ ║╠╣  ║ ║║║╠═╣╠╦╝║╣   ║╣  ║ ╠╦╝║ ║║  ╔═╝
# ╚═╝╚═╝╚   ╩ ╚╩╝╩ ╩╩╚═╚═╝  ╚═╝ ╩ ╩╚═╚═╝╚═╝╚═╝

import numpy as np
import os

from tamalero.utils import load_yaml, ffs, bit_count
from tamalero.ETROC import ETROC
from crcETROC import mod2div

here = os.path.dirname(os.path.abspath(__file__))
maxpixel = 256

class ETROC2_Emulator(ETROC):
    def __init__(self, BCID=0, verbose=False, chipid=123456, elink=0):
        self.isfake = True
        if verbose:
            print('Initiating software ETROC2 (software emulator) ...\n')

        self.connected      = True
        self.master         = "software"
        self.i2c_channel    = "0"
        self.elink          = elink
        self.ver            = "23-2-23"  # yy-mm-dd
        self.rb             = None

        # load ETROC2 dataformat
        self.format = load_yaml(os.path.expandvars('$TAMALERO_BASE/configs/dataformat.yaml'))['ETROC2']

        # load register map
        self.regs = load_yaml(os.path.join(here, '../address_table/ETROC2_example.yaml'))

        # storing data for running L1As
        self.data = {
                'elink'     : elink,
                'l1counter' : 0,
                'bcid'      : BCID,
                'type'      : 0,  # use type regular data
                'chipid'    : chipid,
                'status'    : 0,
                'hits'      : 0,
                'crc'       : 0,
                'vth'       : 198,
                }

        # data from most recent L1A (list of formatted words)
        self.L1Adata = []

        # number of bits in a word
        self.nbits = self.format['nbits']

        # generate fake baseline/noise properties per pixel
        self.bl_means  = [[np.random.normal(700, 2.0) for x in range(16)] for y in range(16)]
        self.bl_stdevs = [[np.random.normal(  1, .2) for x in range(16)] for y in range(16)]

        # this represents the registers on the actual chip
        self.register = {adr: 0 for adr in range(2**16)}  # fill all registers with 0

        self.default_config()

        self.DAC_min  = 600  # in mV
        self.DAC_max  = 1000  # in mV
        self.DAC_step = 400/2**10

    def write_adr(self, adr, val):
        self.register[adr] = val

    def read_adr(self, adr):
        return self.register[adr]

    # add hit data to self.L1Adata & increment hit counter
    def add_hit(self, row, col):
        # generate random data
        data = {
                'ea'     : 0,
                'row_id' : row,
                'col_id' : col,
                'toa'    : np.random.randint(0,500),
                'cal'    : np.random.randint(0,500),
                'tot'    : np.random.randint(0,500),
                }
        # format data
        word = self.format['identifiers']['data']['frame']
        for datatype in data:
            word = ( word +
                ((data[datatype]<<self.format['data']['data'][datatype]['shift'])
                &self.format['data']['data'][datatype]['mask']) )

        if self.data['hits'] < 255:
            # the data format does not allow us to actually have 256 hits
            # so we always have to cut the last one if there would be 100% occupancy
            self.L1Adata.append(word)

            # inc Nhits
            self.data['hits'] += 1

        return None


    # run one L1A
    def runL1A(self):
        self.data['hits'] = 0
        self.L1Adata = [] # wipe previous L1A data
        self.data['l1counter'] += 1

        for row in range(16):
            for col in range(16):
                # produce random hit
                val = np.random.normal(self.bl_means[row][col], self.bl_stdevs[row][col])
                # if we have a hit
                if val > self.get_Vth_mV():
                    self.add_hit(row, col)

        data = self.get_data()
        return data


    # run N L1As and return all data from them
    def run(self, N):
        data = []
        for i in range(N):
            self.runL1A()
            data += self.get_data()
        return data


    # return full data package (list of words) for most recent L1A
    def get_data(self):
        # form header word
        header = self.format['identifiers']['header']['frame']
        for datatype in ['l1counter', 'type', 'bcid']:
            header = ( header +
                ((self.data[datatype]<<self.format['data']['header'][datatype]['shift'])
                &self.format['data']['header'][datatype]['mask']) )

        # form trailer word
        trailer = self.format['identifiers']['trailer']['frame']
        for datatype in ['chipid', 'status', 'hits', 'crc']:
            trailer = ( trailer +
                ((self.data[datatype]<<self.format['data']['trailer'][datatype]['shift'])
                &self.format['data']['trailer'][datatype]['mask']) )

        frame = [header] + self.L1Adata + [trailer]

        #Computing CRC and adding it to the trailer
        poly ='100101111' #crc generator polynomial
        binstr40 = np.vectorize(lambda x: f'{x:040b}') #joining event frames into a string of bits
        merged_frames = "".join(binstr40(frame)) 
        crc_val = mod2div(merged_frames,poly) #compute CRC value
        frame[-1] = trailer +int(crc_val,2) 
        
        return frame