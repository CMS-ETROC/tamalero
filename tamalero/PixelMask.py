#!/usr/bin/env python3
import numpy as np
import yaml
from yaml import Dumper, Loader

class PixelMask:

    def __init__(self, ar=None):
        if ar is not None:
            self.pixels = ar
        else:
            self.pixels = np.ones([16,16])

    @classmethod
    def from_file(cls, f_in):
        with open(f_in, 'r') as f:
            p = yaml.load(f, Loader=Loader)
        return cls(ar=np.array(p))

    def disable_pixels(self, pixels=[], rows=[], cols=[]):
        for row, col in pixels:
            self.pixels[row, col] = 0
        for row in rows:
            self.pixels[row, :] = 0
        for col in cols:
            self.pixels[:, col] = 0

    def get_masked_pixels(self):
        if hasattr(self, 'masked_pixels'):
            return self.masked_pixels
        else:
            self.masked_pixels = []
            for row in range(16):
                for col in range(16):
                    if self.pixels[row, col] == 0:
                        self.masked_pixels.append((row, col))
            return self.masked_pixels

    def dump(self, f_out):
        with open(f_out, 'w') as f:
            yaml.dump(self.pixels.tolist(), f)

    def show(self):
        for row in self.pixels:
                print ((' '.join([ str(int(x)) for x in row])).replace('0', 'O').replace('1', 'X'))
        #pass


if __name__ == '__main__':

    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--pixels', action='store', default=None, help="Individual pixels to mask")
    argParser.add_argument('--rows', action='store', default=None, help="Rows to mask")
    argParser.add_argument('--cols', action='store', default=None, help="Columns to mask")
    argParser.add_argument('--input', action='store', default=None, help="Mask to load")
    argParser.add_argument('--output', action='store', default=None, help="Where to store the mask")
    args = argParser.parse_args()

    if args.input:
        mask = PixelMask.from_file(args.input)
    else:
        mask = PixelMask()

    rows   = []
    cols   = []
    pixels = []
    if args.rows:
        rows = [int(x) for x in args.rows.split(',')]
    if args.cols:
        cols = [int(x) for x in args.cols.split(',')]
    if args.pixels:
        pixels = list(zip([int(x) for x in args.pixels.split(',')[::2]], [int(x) for x in args.pixels.split(',')[1::2]]))

    mask.disable_pixels(rows=rows, cols=cols, pixels=pixels)

    mask.show()

    if args.output:
        mask.dump(args.output)
