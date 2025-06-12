#!/usr/bin/env python3
import os
import awkward as ak
import numpy as np
import json

import hist
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
plt.style.use(hep.style.CMS)

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    mc = LinearSegmentedColormap.from_list("my_colormap", [(1,1,1), (219./255, 58./255, 52./255)])

    all_events = []
    #for i in range(5588,5706):
    for i in range(5707,6106):
    #for i in range(6307,6506):
    #for i in range(5707,6500):
    #for i in range(5707,5708):
        in_file = f"{here}/../ETROC_output/{i}_merged.json"
        if os.path.isfile(in_file):
            with open(in_file, "r") as f:
                all_events.append(ak.from_json(json.load(f)))
        else:
            print(f'Missing file: {in_file}')

    events = ak.concatenate(all_events)
    events['bin'] = 3.125 / events.cal_code
    events['toa'] = 12.5 - events.bin * events.toa_code
    events['x'] = (2*events.row + 1) * 1.3/2 + 0.3  # 300um edge, 1.3mm wide pixels
    events['y'] = (30 - 2*events.col - 1) * 1.3/2 - 0.6  # 300um edge, 1.3mm wide pixels
    events['z'] = -122*(events.chipid==(37<<2)) - 61*(events.chipid==(36<<2))


    all_layer_hit_candidates = events[ak.all(events.nhits==1, axis=1)]
    all_layer_hit_candidates_no_noise_selection = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] < 5))]) >0)

    #((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] == 1) & ((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] == 12)))
    # events[ak.all(events.nhits, axis=1)].toa_code
    #
    #


    hits0 = np.zeros([16, 16])
    hits1 = np.zeros([16, 16])
    hits2 = np.zeros([16, 16])
    for ev in all_layer_hit_candidates[all_layer_hit_candidates_no_noise_selection]:
        for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
            hits0[row,col]+=1
        for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
            hits1[row,col]+=1
        for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
            hits2[row,col]+=1

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    cax = ax[2].matshow(hits0, cmap=mc)
    ax[2].set_title("Module 38")
    cax = ax[1].matshow(hits1, cmap=mc)
    ax[1].set_title("Module 36")
    cax = ax[0].matshow(hits2, cmap=mc)
    ax[0].set_title("Module 37")
    ax[0].set_ylabel(r'$Row$')
    ax[0].set_xlabel(r'$Column$')
    ax[1].set_xlabel(r'$Column$')
    ax[2].set_xlabel(r'$Column$')
    fig.colorbar(cax,ax=ax)
    fig.savefig(f"{here}/../ETROC_output/merged_layers_heatmap_2.pdf")
    fig.savefig(f"{here}/../ETROC_output/merged_layers_heatmap_2.png")




    all_layer_hit_candidates_single_pixel = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] ==1)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] ==12)))]) >0)
    hits0 = np.zeros([16, 16])
    hits1 = np.zeros([16, 16])
    hits2 = np.zeros([16, 16])
    for ev in all_layer_hit_candidates[all_layer_hit_candidates_single_pixel]:
        for row, col in zip(ev.row[ev.chipid==(38 << 2)], ev.col[ev.chipid==(38 << 2)]):
            hits0[row,col]+=1
        for row, col in zip(ev.row[ev.chipid==(36 << 2)], ev.col[ev.chipid==(36 << 2)]):
            hits1[row,col]+=1
        for row, col in zip(ev.row[ev.chipid==(37 << 2)], ev.col[ev.chipid==(37 << 2)]):
            hits2[row,col]+=1


    fig, ax = plt.subplots(1,3,figsize=(15,5))
    cax = ax[2].matshow(hits0, cmap=mc)
    ax[2].set_title("Module 38")
    cax = ax[1].matshow(hits1, cmap=mc)
    ax[1].set_title("Module 36")
    cax = ax[0].matshow(hits2, cmap=mc)
    ax[0].set_title("Module 37")
    ax[0].set_ylabel(r'$Row$')
    ax[0].set_xlabel(r'$Column$')
    ax[1].set_xlabel(r'$Column$')
    ax[2].set_xlabel(r'$Column$')
    fig.colorbar(cax,ax=ax)
    fig.savefig(f"{here}/../ETROC_output/merged_layers_heatmap_single_pixel.pdf")
    fig.savefig(f"{here}/../ETROC_output/merged_layers_heatmap_single_pixel.png")


    time_axis = hist.axis.Regular(100, -15, 15, name="time", label="time")
    dt_01_hist = hist.Hist(time_axis)
    dt_12_hist = hist.Hist(time_axis)
    dt_02_hist = hist.Hist(time_axis)

    sel_events = all_layer_hit_candidates[all_layer_hit_candidates_no_noise_selection]
    dt_01 = sel_events.toa[sel_events.chipid==(36<<2)] - sel_events.toa[sel_events.chipid==(38<<2)]
    dt_02 = sel_events.toa[sel_events.chipid==(37<<2)] - sel_events.toa[sel_events.chipid==(38<<2)]
    dt_12 = sel_events.toa[sel_events.chipid==(37<<2)] - sel_events.toa[sel_events.chipid==(36<<2)]
    dt_01_hist.fill(ak.flatten(dt_01))
    dt_02_hist.fill(ak.flatten(dt_02))
    dt_12_hist.fill(ak.flatten(dt_12))

    fig, ax = plt.subplots(1,2,figsize=(10,5))

    dt_01_hist.plot1d(ax=ax[1])
    dt_12_hist.plot1d(ax=ax[0])

    ax[1].set_title("36 - 38")
    ax[0].set_title("37 - 36")
    ax[0].set_ylabel(r'$Events$')
    ax[0].set_xlabel(r'$\Delta t\ (ns)$')
    ax[1].set_xlabel(r'$\Delta t\ (ns)$')
    fig.savefig(f"{here}/../ETROC_output/merged_layers_delta_t.pdf")
    fig.savefig(f"{here}/../ETROC_output/merged_layers_delta_t.png")

    # do some simple alignment
    single_pixel = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] ==0)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] ==10)))]) >0)
    sel_events = all_layer_hit_candidates[single_pixel]
    x_ref = np.mean(ak.flatten(sel_events.x[sel_events.chipid==(38<<2)]))
    y_ref = np.mean(ak.flatten(sel_events.y[sel_events.chipid==(38<<2)]))

    x_corr_2 = x_ref - ak.mean(sel_events.x[sel_events.chipid==(36<<2)])
    y_corr_2 = y_ref - ak.mean(sel_events.y[sel_events.chipid==(36<<2)])

    x_corr_3 = x_ref - ak.mean(sel_events.x[sel_events.chipid==(37<<2)])
    y_corr_3 = y_ref - ak.mean(sel_events.y[sel_events.chipid==(37<<2)])


    # apply alignment to different pixel
    single_pixel = (ak.num(all_layer_hit_candidates.col[((all_layer_hit_candidates.row[all_layer_hit_candidates.chipid==(38<<2)] ==1)&((all_layer_hit_candidates.col[all_layer_hit_candidates.chipid==(38<<2)] ==12)))]) >0)
    sel_events = all_layer_hit_candidates[single_pixel]


    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = sel_events.x
    xs = xs + sel_events.chipid==(38<<2)*0
    xs = xs + sel_events.chipid==(36<<2)*x_corr_2
    xs = xs + sel_events.chipid==(37<<2)*x_corr_3

    ys = sel_events.y
    ys = ys + sel_events.chipid==(38<<2)*0
    ys = ys + sel_events.chipid==(36<<2)*y_corr_2
    ys = ys + sel_events.chipid==(37<<2)*y_corr_3

    xs = ak.flatten(xs)
    ys = ak.flatten(ys)
    zs = ak.flatten(sel_events.z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.Polygon([(0.3,0.6), (0.3, 1.9), (1.6,1.9), (1.6,0.6)], fill=True, closed=True, edgecolor='black')

    ax.scatter(xs[0:3], ys[0:3], zs[0:3], c='r')
    ax.scatter(xs[3:6]+0.5, ys[3:6], zs[3:6], c='b')
    ax.scatter(xs[6:9], ys[6:9]+0.5, zs[6:9], c='g')
    ax.scatter(xs[9:12]+0.5, ys[9:12]+0.5, zs[9:12], c='orange')

    ax.set_xlim(0,21.4)
    ax.set_ylim(0,21.4)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    fig.savefig(f"{here}/../ETROC_output/hits_aligned.pdf")
    fig.savefig(f"{here}/../ETROC_output/hits_aligned.png")

    #plt.show()


    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    ## For each set of style and range settings, plot n random points in the box
    ## defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #xs = ak.flatten(sel_events.x)
    #ys = ak.flatten(sel_events.y)
    #zs = ak.flatten(sel_events.z)
    #ax.scatter(xs[:10], ys[:10], zs[:10])

    #ax.set_xlabel('x (mm)')
    #ax.set_ylabel('y (mm)')
    #ax.set_zlabel('z (mm)')

    #plt.show()





    sel_events = all_layer_hit_candidates[all_layer_hit_candidates_no_noise_selection]



    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = ak.flatten(sel_events.x)
    ys = ak.flatten(sel_events.y)
    zs = ak.flatten(sel_events.z)
    ax.scatter(xs[:10], ys[:10], zs[:10])

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    fig.savefig(f"{here}/../ETROC_output/hits_original.pdf")
    fig.savefig(f"{here}/../ETROC_output/hits_original.png")
    #plt.show()
