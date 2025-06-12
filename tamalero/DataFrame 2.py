#!/usr/bin/env python3
import os

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

here = os.path.dirname(os.path.abspath(__file__))

class DataFrame:
    def __init__(self, version='ETROC2'):
        with open(os.path.expandvars(f'{here}/../configs/dataformat.yaml'), 'r') as f:
            self.format = load(f, Loader=Loader)[version]
        self.type = 0

    def get_bytes(self, word, format):
        bytes = []
        if self.format['bitorder'] == 'normal':
            shifts = [32, 24, 16, 8, 0]
        elif self.format['bitorder'] == 'reversed':
            shifts = [0, 8, 16, 24, 32]
        for shift in shifts:
            bytes.append((word >> shift) & 0xFF)
        if format:
            return [ '{0:0{1}x}'.format(b,2) for b in bytes ]
        else:
            return bytes

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
                #print ("Found:", id)
                break
        #print (val, data_type)
        #if data_type == 'filler': print(val)
        res = {}
        if data_type == None:
            if not quiet:
                print ("Found data of type None:", val)
            return None, res

        if data_type == 'data':
            try:
                datatypelist = self.format['types'][self.type]
            except KeyError:
                datatypelist = self.format['types'][0]  # assume a default
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
        #print (res)
        return data_type, res

if __name__ == '__main__':

    test_words = [
        259251636582,
        259243266812,
        576704960435,
        459347591436,
        259251651343,
        259251651348,
        ]

    df = DataFrame('ETROC2')
    for word in test_words:
        df.read(word, quiet=False)
