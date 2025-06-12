#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import glob
import os

if __name__ == '__main__':

    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--filename', action='store', default=False, help="Specify file to plot")
    args = argParser.parse_args()

    if args.filename:
        with open('eye_scan_results/%s' %args.filename, 'r') as openfile:
            eye_data = json.load(openfile)
        print("Plotting %s..." %args.filename)
    else:
        list_of_files = glob.glob('eye_scan_results/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r') as openfile:
            eye_data = json.load(openfile)
        print("Plotting %s..." %latest_file)

    (fig, axs) = plt.subplots(1, 1, figsize=(10, 8))
    print ("fig type = " + str(type(fig)))
    print ("axs type = " + str(type(axs)))
    axs.set_title("LpGBT 2.56 Gbps RX Eye Opening Monitor")
    plot = axs.imshow(eye_data, alpha=0.9, vmin=0, vmax=100, cmap='jet',interpolation="nearest", aspect="auto",extent=[-384.52/2,384.52/2,-0.6,0.6,])
    plt.xlabel('ps')
    plt.ylabel('volts')
    fig.colorbar(plot, ax=axs)

    #plt.show()
    fig.savefig('eye_scan_results/eye.png')
