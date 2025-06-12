import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
from scipy.optimize import curve_fit
import os
import sys
import pdb
import pandas as pd


board_number     = 40
# base             = "/home/daq/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged"
base             = "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged"
# base = "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/ScopeData/ETROCData"
# base = "/home/etl/Test_Stand/ETROC2_Test_Stand/ScopeHandler/Lecroy/Merging"
legend_label     = "Threshold offset"
scanning_version = "Threshold_offset"
output_folder    = f"p_no_cuts/plots_{scanning_version}_N{board_number}_358_one_channel"
unit             = "vth"
os.system(f"mkdir -p {output_folder}")

y_axis = [358]
x_axis = [8]

PIXELS = [0]

print(len(x_axis))
print(len(y_axis))

a = 10
c = 1
do_Delta_T_Gaussian_fit = True

# argParser.add_argument('--one_pixel', action = 'store', default = False, type = bool, help = 'One pixel flag')
# argParser.add_argument('--row', action = 'store', type = int, default = 8, help = 'Pixel row')
# argParser.add_argument('--col', action='store', type = int, default = 8, help = "Pixel column")

def merge_unique(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    if len(merged_list) != 0:
        merged_list = len(set(merged_list))
    else:
        merged_list = len([])
    return merged_list

def get_unique(row, col):
    pairs = ak.Array([merge_unique(row[i], col[i]) for i in range(len(row))])
    return np.array(pairs)

def load_data(indices):
    output = []
    for i, index in enumerate(indices):
        filename = f"{base}/run_{y_axis[i]}.root"
        output.append({})
        with uproot.open(filename) as f:
            tree = f["pulse"]
            output[i]["nhits"] = tree["nhits"].array()
            tbin               = 3.125 / tree["cal_code"].array()
            output[i]["TOA"]   = fill_empty_arrays(12.5 - tbin * tree["toa_code"].array())
            output[i]["TOT"]   = fill_empty_arrays((2 * tree["tot_code"].array() - np.floor(tree["tot_code"].array() / 32)) * tbin)
            output[i]["CAL"]   = fill_empty_arrays(tree["cal_code"].array())
            output[i]["row"]   = fill_empty_arrays(tree["row"].array())
            output[i]["col"]   = fill_empty_arrays(tree["col"].array())
            # output[i]["DT"] = output[i]["TOA"] - (tree[f"Clock"].array() - tree[f"LP2_1"].array()[:, 1])
            
            # Get the timestamps
            for c in [1, 2]:                        # The index of the scope channel
                for p in [20]: # The percentage of amplitude value
                    if c == 2:
                        output[i]["Clock"] = tree["Clock"].array() # tree[f"linear_RE_{p}"].array()[:, c]
                    if c == 1:
                        output[i][f"LP2_{p}_{c}"] = tree[f"LP2_{p}"].array()[:, c]
    return output

def gaussian(x, a, mean, sigma):
    return (a/np.sqrt(2 * np.pi * sigma)) * np.exp(-(x - mean)**2 / (2 * sigma**2))

def chi_squared(obs, pred):
    diff = [((obs[i] - pred[i])**2 / pred[i]) for i in range(len(pred))]
    print(sum(diff))
    print(sum(diff)/len(diff))
    return np.sum(diff)

def fit_curve(data, function, hist, bounds = (-np.inf, np.inf)):
    data = np.array(data)
    mean = np.mean(data)
    rms = np.std(data)
    x = (hist[1] + (hist[1][1] - hist[1][0]) / 2)[:-1]
    # hist[0] = hist[0][(x > bounds[0]) & (x < bounds[1])]
    return curve_fit(function, x[(x > bounds[0]) & (x < bounds[1])], hist[0][(x > bounds[0]) & (x < bounds[1])], p0 = [0.01, mean, rms], absolute_sigma = True)

def fill_empty_arrays(data, fill_width = -1):
    new_data = []
    print(np.sum(ak.count(data, axis = 1) == 0))
    # np.asarray(data)[ak.count(data, axis = 1) == 0] = [-1]
    for i in range(len(data)):
        if len(data[i]) != 0:
            new_data.append(list(data[i]))
        else:
            new_data.append([-1])
    # ak.insert(data[ak.count(data, axis = 1) == 0], i, -1)
    print(ak.Array(new_data))
    return ak.Array(new_data)

def remove_empty(data):
    map = ak.count(data, axis = -1) > 0
    data = data[map]
    return data

def rescale(elem):
    if elem < 0:
        elem = elem - 25 * np.floor(elem / 25)
    elif elem > 25:
        elem = elem - 25 * np.ceil(elem / 25)
    return elem

def scan(data): # a dictionary
    print(list(data.keys()))
    print(len(data[list(data.keys())[0]]), len(data[list(data.keys())[1]]), len(data[list(data.keys())[2]]))
    maximum = max([len(data[list(data.keys())[0]]), len(data[list(data.keys())[1]]), len(data[list(data.keys())[2]])])
    if not (len(data[list(data.keys())[0]]) == len(data[list(data.keys())[1]]) == len(data[list(data.keys())[2]])):
        for i in range(maximum):
            for j in data.keys():
                print(data[j][i], end = ",")
            print()

def remove_empty_arrays(array):
    return array[ak.count(array, axis = 1) != 0]

def plots(data, x_axis):
    mean_toa = [[] for i in range(4)]
    mean_tot = [[] for i in range(4)]
    mean_cal = [[] for i in range(4)]
    std_toa  = [[] for i in range(4)]
    std_tot  = [[] for i in range(4)]
    std_cal  = [[] for i in range(4)]
    std_clock   = [[] for i in range(4)]
    std_trigger = [[] for i in range(4)]
    mean_DT  = [[] for i in range(4)]
    std_DT   = [[] for i in range(4)]
    TOA      = [[] for i in range(4)]
    TOT      = [[] for i in range(4)]
    CAL      = [[] for i in range(4)]
    DT       = [[] for i in range(4)]
    hits     = []
    hits_    = [[] for i in range(len(x_axis))]
    Clock_   = [[] for i in range(len(x_axis))]
    LP2_20_7_= [[] for i in range(len(x_axis))]
    hit_rate = []
    for d_i, d in enumerate(data):
        cutflow_tables = {}
        # -------------------------- EVENT LEVEL SELECTION --------------------------
        sel = np.ones(len(d["nhits"]), dtype = bool) # ((d["nhits"] > -1)) #  & (d["Clock"] < 0)) # Select events with at least one hit.
        row = d["row"][sel]
        col = d["col"][sel]
        toa = d["TOA"][sel]
        tot = d["TOT"][sel]
        cal = d["CAL"][sel]
        Clock_[d_i] = d["Clock"][sel] # np.array([rescale(i) for i in d["Clock"][sel]])
        LP2_20_7_[d_i] = d["LP2_20_1"][sel]*10**9
        hits_[d_i] = d["nhits"][sel]

        elem_sel  = ((cal > 150) & (cal < 210) & (tot > 2.0) & (tot < 15.0)) # Select hits from a particular pixel.

        Clock_   [d_i] = Clock_   [d_i][ak.any(elem_sel, axis = 1)]
        LP2_20_7_[d_i] = LP2_20_7_[d_i][ak.any(elem_sel, axis = 1)]
        hits_    [d_i] = hits_    [d_i][ak.any(elem_sel, axis = 1)]
        hit_rate.append(len(hits_[d_i]) / 10000)
        row = row[elem_sel]
        col = col[elem_sel]
        toa = toa[elem_sel]
        tot = tot[elem_sel]
        cal = cal[elem_sel]

        row = remove_empty_arrays(row)
        col = remove_empty_arrays(col)
        tot = remove_empty_arrays(tot)
        cal = remove_empty_arrays(cal)
        toa = remove_empty_arrays(toa)
        p = 0
        for p_i in PIXELS:
            cuts = {}
            # -------------------------- ELEMENT LEVEL SELECTION --------------------------
            pixel_sel = ((row > 0) & (col > 0))
            print(len(pixel_sel), len(Clock_[d_i]))
            print(f"Row: 8, Col: {p_i}")
            row_ = row[(pixel_sel)]
            col_ = col[(pixel_sel)]
            toa_ = toa[(pixel_sel)]
            tot_ = tot[(pixel_sel)]
            cal_ = cal[(pixel_sel)]
            obj = {"TOA": toa_, "TOT": tot_, "CAL": cal_}
            # event_sel = (LP2_20_6 != 10000)
            # row_ = row_[event_sel]
            # col_ = col_[event_sel]
            # toa_ = toa_[event_sel]
            # tot_ = tot_[event_sel]
            # cal_ = cal_[event_sel]
            # element_sel = ((cal_ > 195) & (cal_ < 205)) # Other object level selections.
            # toa_ = toa_[element_sel]
            # tot_ = tot_[element_sel]
            # cal_ = cal_[element_sel]
            # maps = (ak.count(, axis = 1) != 0)
            maps = (ak.any(pixel_sel, axis = 1))
            row_ = remove_empty_arrays(row_)[:,0]
            col_ = remove_empty_arrays(col_)[:,0]
            tot_ = remove_empty_arrays(tot_)[:,0]
            cal_ = remove_empty_arrays(cal_)[:,0]
            toa_ = remove_empty_arrays(toa_)[:,0]
            mean_toa   [p].append(np.mean(toa_))
            mean_tot   [p].append(np.mean(tot_))
            mean_cal   [p].append(np.mean(cal_))
            TOA        [p].append(toa_)
            TOT        [p].append(tot_)
            CAL        [p].append(cal_)
            std_toa    [p].append(np.std (toa_))
            std_tot    [p].append(np.std (tot_))
            std_cal    [p].append(np.std (cal_))
            std_clock  [p].append(np.std (Clock_[d_i][maps]))
            std_trigger[p].append(np.std (LP2_20_7_[d_i][maps]))
            DT         [p].append(toa_ - (ak.Array(LP2_20_7_[d_i][maps]) - ak.Array(Clock_[d_i][maps])))
            mean_DT    [p].append(np.mean(DT[p][d_i]))
            std_DT     [p].append(np.std (DT[p][d_i]))
            cutflow_tables[f"row_15_col_{p}"] = cuts
            p+=1

    # Plot graphs.
    '''
    variables = [mean_toa, mean_tot, mean_cal, std_toa, std_tot, std_cal, std_clock, std_trigger, mean_DT, std_DT]
    variables = [np.nan_to_num(var) for var in variables]
    labels    = ["mean_toa", "mean_tot", "mean_cal", "std_toa", "std_tot", "std_cal", "std_clock", "std_trigger", "mean_DT", "std_DT"]
    vars      = dict(zip(labels, variables))
    for v, var in enumerate(variables):
        plt.style.use(hep.style.CMS)
        fig, ax = plt.subplots(1, 1, figsize = (1*a*c, 1*a))
        hep.cms.label(llabel="ETL Preliminary", rlabel="")
        for p in range(4):
            var = np.array(var)
            var[var==np.inf] = 0
            if "mean" in labels[v]:
                ax.errorbar(x_axis, var[p], yerr = vars["std" + labels[v].split("mean")[1]][p], label = f"Row: 15, Col: {p}", fmt = "o", capsize = 50)
            else:
                ax.scatter(x_axis, var[p], label = f"Row: 15, Col: {p}", s=100)
            if "std_DT" == labels[v]:
                print("Row: 15, Col: ",p ,var[p])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        ax.set_xlabel(f"{legend_label} [{unit}]", loc='center')
        ax.set_ylabel(labels[v],loc='center')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{output_folder}/{labels[v]}.png")
        plt.close()
    '''
    # Plot histograms.
    '''
    variables = [TOA  ,  TOT ,  CAL , \
                 LP2_20_6_ ,  LP2_30_6_ ,  LP2_35_6_ ,  LP2_40_6_ , LP2_50_6_  ,  LP2_60_6_ ,  LP2_70_6_ ,  LP2_80_6_ , \
                 LP2_20_7_ ,  LP2_30_7_ ,  LP2_35_7_ ,  LP2_40_7_ , LP2_50_7_  ,  LP2_60_7_ ,  LP2_70_7_ ,  LP2_80_7_ ,   hits_ ,   DT]

    labels    = ["TOA", "TOT", "CAL", \
                 "LP2_20_6_", "LP2_30_6_", "LP2_35_6_", "LP2_40_6_", "LP2_50_6_", "LP2_60_6_", "LP2_70_6_", "LP2_80_6_", \
                 "LP2_20_7_", "LP2_30_7_", "LP2_35_7_", "LP2_40_7_", "LP2_50_7_", "LP2_60_7_", "LP2_70_7_", "LP2_80_7_", "hits", "DT"]
    '''
    print(np.array(x_axis)+88)
    print(hit_rate)
    plt.plot(np.array(x_axis)+88, hit_rate)
    plt.ylabel("Fraction of events")
    plt.xlabel("Threshold (Baseline + offset) [vth]")
    plt.show()
    
    variables = [TOA  ,  TOT ,  CAL , \
                 Clock_ , \
                 LP2_20_7_ ,  hits_ ,   DT]

    labels    = ["TOA", "TOT", "CAL", \
                 "Clock", \
                 "Trigger", "hits", "DT"]

    units     = ["ns", "ns", "ns", "ns", "ns", "", "ns"]
    vars      = dict(zip(labels, variables))

    for v, var in enumerate(variables):
        if labels[v] == "TOA" or labels[v] == "TOT" or labels[v] == "CAL" or labels[v] == "DT":
            fig, ax = plt.subplots(2, 2, figsize = (2*a*c, 2*a))
            for p in range(len(PIXELS)):
                minimum = min([min(var[p][i]) if len(var[p][i]) != 0 else 10000 for i in range(len(x_axis))])
                maximum = max([max(var[p][i]) if len(var[p][i]) != 0 else 0     for i in range(len(x_axis))])
                x = int(np.floor(p / 2))
                y = int(p % 2)
                for j in range(len(x_axis)):
                    if (minimum - maximum) != 10000:
                        # print(minimum, maximum)
                        bins = np.linspace(minimum, maximum, 70)
                    else:
                        bins = 70
                    # bins = np.linspace(minimum, maximum, 70)
                    if labels[v] == "DT":
                        minimum = 5
                        maximum = 15
                    weights = np.ones(len(var[p][j])) / len(var[p][j])
                    ax[x][y].hist(var[p][j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                ax[x][y].set_xlabel(f"{labels[v]} [{units[v]}]",loc='center')
                ax[x][y].set_ylabel("Events",loc='center')
                ax[x][y].set_title(f"Row: 15, Col: {p}")
                ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
                handles, label = ax[x][y].get_legend_handles_labels()
                idx = np.sort(np.unique(np.array(label), return_index=True)[1])
                ax[x][y].legend(np.array(label)[idx])
            plt.show()
            fig.savefig(f"{output_folder}/{labels[v]}_HIST.png")
            plt.close()
        else:
            plt.style.use(hep.style.CMS)
            fig, ax = plt.subplots(1, 1, figsize = (1*a*c, 1*a))
            hep.cms.label(llabel="ETL Preliminary", rlabel="")
            minimum = min([min(var[i]) if len(var[i]) != 0 else 10000 for i in range(len(x_axis))])
            maximum = max([max(var[i]) if len(var[i]) != 0 else 0     for i in range(len(x_axis))])
            for j in range(len(x_axis)):
                if (minimum - maximum) != 10000:
                    # print(minimum, maximum)
                    bins = np.linspace(minimum, maximum, 70)
                else:
                    bins = 70
                weights = np.ones(len(var[j])) / len(var[j])
                if labels[v] == 'hits':
                    bins = 5
                    ran = (0,5)
                    ax.hist(var[j], label = f"{legend_label}: {x_axis[j]} {unit}",range=ran, bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                    ax.set_xlim(0,5)
                else:
                    ax.hist(var[j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
            ax.set_xlabel(f"{labels[v]} [{units[v]}]",loc='center')
            ax.set_ylabel("Events",loc='center')
            plt.show()
            handles, label = ax.get_legend_handles_labels()
            idx = np.sort(np.unique(np.array(label), return_index=True)[1])
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            ax.legend(np.array(label)[idx])
            fig.savefig(f"{output_folder}/{labels[v]}_HIST.png")
            plt.tight_layout()
            plt.close()

    for i in range(len(x_axis)): # toa vs tot
        fig, ax = plt.subplots(2, 2, figsize = (2*a*c, 2*a))
        for p in range(len(PIXELS)):
            x = int(np.floor(p / 2))
            y = int(p % 2)
            ax[x][y].hist2d(np.array(variables[0][p][i]), np.array(variables[1][p][i]), bins = (50, 50))
            ax[x][y].set_title(f"Row: 15, Col: {p}")
            ax[x][y].set_xlabel(f"TOA [{units[0]}]", loc='center')
            ax[x][y].set_ylabel(f"TOT [{units[1]}]", loc='center')
            ax[x][y].set_xlim([2,12.5])
            ax[x][y].set_ylim([0,15])
            ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
        plt.show()
        fig.savefig(f"{output_folder}/TOA_vs_TOT_row_15_col_{p}_{x_axis[i]}.png")

    # for i in range(len(x_axis)): # tot vs Clock
    #     fig, ax = plt.subplots(1, 1, figsize = (2*a*c, 2*a))
    #     ax.hist2d(np.array(variables[3][i]), np.array(variables[1][i]), bins = (50, 50))
    #     ax.set_title(f"Row: 15, Col: {p}")
    #     ax.set_xlabel("nHits",loc='center')
    #     ax.set_ylabel("Clock",loc='center')
    #     ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    #     fig.savefig(f"{output_folder}/TOT_Clock_{x_axis[i]}.png")

    y_axis = [10985, 10987, 10990]
    x_axis = [10,14,20]

    if do_Delta_T_Gaussian_fit:
        fig, ax = plt.subplots(2, 2, figsize = (2*20*c, 2*20))
        for p in range(len(PIXELS)):
            x = int(np.floor(p / 2))
            y = int(p % 2)
            bins = np.linspace(-3, -2, 55)
            for j in range(len(x_axis)):
                mean_val = np.mean(DT[p][j])
                weights = np.ones(len(DT[p][j])) / len(DT[p][j])
                minimum = mean_val - 2 * np.std(DT[p][j])
                maximum = mean_val + 0.05 # * np.std(DT[p][j])
                # std_times_5 = 3 * np.std(DT[p][j])
                hist = ax[x][y].hist(DT[p][j], label = f"{legend_label}: {x_axis[j]} {unit}", bins = bins, histtype = "step", weights = weights, linewidth=2.5)
                if len(DT[p][j]) < 10: continue
                minimum_fit = mean_val - 0.25 * np.std(DT[p][j])
                maximum_fit = mean_val + 2 * np.std(DT[p][j])
                curve = fit_curve(DT[p][j], gaussian, hist, bounds = (minimum_fit, maximum_fit))
                mean = np.mean(DT[p][j])
                x_range = np.array((hist[1] + (hist[1][1] - hist[1][0]) / 2))
                y_range = hist[0]
                x_range = x_range[0:len(y_range)]
                # delta = np.abs(x_range - mean)
                # sigma = np.std(delta)
                # x_range = x_range[delta < 1.0*sigma]
                # y_range = y_range[delta < 1.0*sigma]
                chi = np.round(chi_squared(y_range, gaussian(x_range, *curve[0])) / len(y_range), 2)
                ax[x][y].plot(x_range[(x_range > minimum_fit) & (x_range < maximum_fit)], gaussian(x_range, *curve[0])[(x_range > minimum_fit) & (x_range < maximum_fit)], "r", linewidth=2, label = "Gaussian fit\n$\sigma$ = "+str(round(curve[0][2]*1000,2))+" ps"+"\n $\chi^{2}$/N = "+str(chi)+".")
            ax[x][y].set_xlabel("$\Delta$T (ps)",loc='center')
            ax[x][y].set_ylabel("Events",loc='center')
            ax[x][y].set_title(f"Row: 15, Col: {p}")
            ax[x][y].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            handles, label = ax[x][y].get_legend_handles_labels()
            idx = np.sort(np.unique(np.array(label), return_index=True)[1])
            ax[x][y].legend(np.array(label)[idx])
        fig.savefig(f"{output_folder}/{labels[v]}_HIST_FIT.png")
        plt.close()

if __name__ == "__main__":
    data = load_data(y_axis)
    plots(data, x_axis)
