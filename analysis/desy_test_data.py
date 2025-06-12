#!/usr/bin/env python3
import numpy as np
import awkward as ak
import uproot
import glob
import os
import pandas as pd
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import norm
import argparse
plt.style.use(hep.style.CMS)

from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore")

def gaus(x, *p0):
    N, mu, sigma = p0
    return N*np.exp(-(x-mu)**2/(2.0*sigma**2))

def linear(x, *p0):
    a, b = p0
    return a + b*x

def polynomial(x, *p0):
    a, b, c, d = p0
    return a + b*x + c*x**2 + d*x**3

if __name__ == '__main__':

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_start', action='store', default=2737, help="Run to start with")
    argParser.add_argument('--run_stop',  action='store', default=3236, help="Run to start with")
    argParser.add_argument('--hitmap', action='store_true', help="Stop after the hitmap")
    argParser.add_argument('--unbinned', action='store_true', help="Run an unbinned fit for time walk corrections")
    argParser.add_argument('--polynomial', action='store_true', help="Run a poly fit instead of linear")
    argParser.add_argument('--lp2', action='store', default=20, choices=['5', '10', '15', '20', '25', '30', '35', '40', '50', '60', '70', '80'], help="Threshold to use for trigger pulse")
    args = argParser.parse_args()

    step_size = 100000

    run_start = int(args.run_start)
    run_stop = int(args.run_stop)
    post_fix = ''
    if args.unbinned:
        post_fix += '_unbinned'
    if args.polynomial:
        post_fix += '_polynomial'
    plot_dir = f'daniel_tmp/start_{run_start}_stop_{run_stop}_LP2_{args.lp2}{post_fix}/'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    all_files = [f"../../ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/run_{r}.root" for r in range(run_start, run_stop+1)]
    in_files = []
    for f in all_files:
        if os.path.isfile(f):
            in_files.append(f)

    print ("Getting hit matrix and mean cal values")
    cal_sum = np.zeros((16,16))
    hit_matrix = np.zeros((16,16))
    for events in uproot.iterate(in_files, step_size=step_size):
        scope_sel = ((events.amp[:,1] > 50) & (events.amp[:,1] < 300))
        toa_sel = ((events.toa_code>100) & (events.toa_code<500))
        tot_sel = (events.tot_code>50)
        for row in range(16):
            for col in range(16):
                pix_sel = ((events.row==row)&(events.col==col))
                sel = pix_sel & scope_sel & toa_sel & tot_sel
                cal_sum[row][col] += ak.sum(events.cal_code[sel])
                hit_matrix[row][col] += ak.count(events.cal_code[sel])

    cal_mean = cal_sum / hit_matrix


    fig, ax = plt.subplots(1,1,figsize=(15,15))
    cax = ax.matshow(hit_matrix)
    ax.set_ylabel(r'$Row$')
    ax.set_xlabel(r'$Column$')
    fig.colorbar(cax,ax=ax)
    for i in range(16):
        for j in range(16):
            text = ax.text(j, i, int(hit_matrix[i,j]),
                    ha="center", va="center", color="w", fontsize="xx-small")
    fig.savefig(f"{plot_dir}/hit_matrix_run_start_{run_start}_run_stop_{run_stop}.pdf")
    fig.savefig(f"{plot_dir}/hit_matrix_run_start_{run_start}_run_stop_{run_stop}.png")


    fig, ax = plt.subplots(1,1,figsize=(15,15))
    cax = ax.matshow(cal_mean)
    ax.set_ylabel(r'$Row$')
    ax.set_xlabel(r'$Column$')
    fig.colorbar(cax,ax=ax)
    for i in range(16):
        for j in range(16):
            text = ax.text(j, i, int(np.nan_to_num(cal_mean[i,j],0)),
                    ha="center", va="center", color="w", fontsize="xx-small")
    fig.savefig(f"{plot_dir}/cal_mean_run_start_{run_start}_run_stop_{run_stop}.pdf")
    fig.savefig(f"{plot_dir}/cal_mean_run_start_{run_start}_run_stop_{run_stop}.png")

    if args.hitmap:
        exit()

    print ("Filling histograms")

    #print("Loading files")
    #events = uproot.concatenate(in_files)
    ##test_file = uproot.open("../ETROC2_Test_Stand/ScopeHandler/ScopeData/LecroyMerged/run_606.root")
    ##events = test_file['pulse'].array()

    #print("Done")
    time_axis   = hist.axis.Regular(100, 10, 15, name="time", label="time")
    dt_axis     = hist.axis.Regular(100, -2, 2, name="time", label="time")
    toa_axis    = hist.axis.Regular(100, 0, 12.5, name="toa", label="toa")
    code_axis   = hist.axis.Regular(50, 100, 200, name="code", label="code")
    cal_axis    = hist.axis.Integer(0,1024, name='cal', label='CAL')
    trigger_axis = hist.axis.Regular(100, -5, 0, name="time", label="time")
    clock_axis = hist.axis.Regular(100, -12.5, 12.5, name="time", label="time")
    row_axis    = hist.axis.Integer(0, 16, name='row', label='Row')
    col_axis    = hist.axis.Integer(0, 16, name='col', label='Row')
    tot_axis    = hist.axis.Regular(10, 3.5, 6.0, name="tot", label="tot")
    #row_cat     = hist.axis.IntCategory([], name="row", label="Dataset", growth=True)
    #col_cat     = hist.axis.IntCategory([], name="col", label="Dataset", growth=True)

    # TOA Hist
    toa_hist = hist.Hist(toa_axis, row_axis, col_axis)
    tot_hist = hist.Hist(toa_axis, row_axis, col_axis)
    cal_hist = hist.Hist(code_axis, row_axis, col_axis)
    cal_hist2 = hist.Hist(cal_axis)
    cal_hist3 = hist.Hist(cal_axis)

    dt_tot_hist = hist.Hist(time_axis, tot_axis, row_axis, col_axis)

    trigger_hist    = hist.Hist(trigger_axis)
    clock_hist      = hist.Hist(clock_axis)
    dt_hist         = hist.Hist(time_axis, row_axis, col_axis)
    dt_hist2        = hist.Hist(time_axis, row_axis, col_axis)
    dt_hist3        = hist.Hist(dt_axis, row_axis, col_axis)

    tw_corr_0 = np.zeros((16,16))
    tw_corr_1 = np.zeros((16,16))
    tw_corr_2 = np.zeros((16,16))
    tw_corr_3 = np.zeros((16,16))

    lp_val = f'LP2_{args.lp2}'

    x = []
    y = []

    res = np.zeros([16, 16])
    for events in uproot.iterate(in_files, step_size=step_size):

        events['bin'] = 3.125 / np.maximum(events.cal_code, 1)
        events['toa'] = 12.5 - events.bin * events.toa_code
        events['tot'] = (2*events.tot_code - np.floor(events.tot_code/32))*events.bin

        events['trigger'] = events[lp_val][:,1] * 10**9
        #events['trigger'] = events['baseline'][:,1]

        events['dt'] = events['toa'] - (events['trigger'] - events['Clock'])


        toa_hist.fill(
            toa = ak.flatten(events['toa']),
            row = ak.flatten(events['row']),
            col = ak.flatten(events['col']),
        )

        tot_hist.fill(
            toa = ak.flatten(events['tot']),
            row = ak.flatten(events['row']),
            col = ak.flatten(events['col']),
        )

        # CAL
        cal_hist.fill(
            code = ak.flatten(events['cal_code']),
            row = ak.flatten(events['row']),
            col = ak.flatten(events['col']),
        )

        # Trigger hist
        scope_sel = ((events.amp[:,1] > 50) & (events.amp[:,1] < 300))
        trigger_hist.fill(
            time = events['trigger'][((scope_sel)&(events.nhits>0))],
        )

        dt_hist.fill(
            time=ak.flatten(events['dt']),
            row=ak.flatten(events['row']),
            col=ak.flatten(events['col']),
        )


        for row in range(16):
            x.append([])
            y.append([])
            for col in range(16):
                x[-1].append([])
                y[-1].append([])
                #print(f'Running {row=} {col=}')
                # in order to remove (noise) tails: need to find CAL code peak
                # remove events with CAL codes away from peak
                scope_sel = ((events.amp[:,1] > 50) & (events.amp[:,1] < 300))
                pix_sel = ((events.row==row)&(events.col==col))
                #cal_hist2.fill(
                #    cal=ak.flatten(events.cal_code[pix_sel])
                #)
                #cal_peak = np.where(cal_hist2.values()==np.max(cal_hist2.values()))[0][0]
                cal_peak = cal_mean[row][col]
                #print(row, col, cal_peak)
                cal_sel = ((events.cal_code<(cal_peak+2)) & (events.cal_code>(cal_peak-2)))
                toa_sel = ((events.toa_code>100) & (events.toa_code<500))
                tot_sel = (events.tot_code>90)
                sel = pix_sel & cal_sel & (scope_sel) & toa_sel & tot_sel

                #cal_mean = ak.mean(events.cal_code[sel])
                #cal = events.cal_code
                time_bin = 3.125 / cal_mean[row][col]
                toa = 12.5 - time_bin * events.toa_code
                tot = ((2*events.tot_code - np.floor(events.tot_code/32))*time_bin)[sel]
                dt = (toa - (events.trigger - events.Clock))[sel]

                #res = np.std(dt)
                #dt_.append({
                #    'row': row,
                #    'col': col,
                #    'dt': np.mean(dt),
                #    'res': res,
                #})
                dt_tot_hist.fill(
                    row=row,
                    col=col,
                    time=ak.flatten(dt),
                    tot=ak.flatten(tot),
                )

                dt_hist2.fill(
                    row=row,
                    col=col,
                    time=ak.flatten(dt),
                )

                if args.unbinned and False:
                    x[-1][-1] += list(tot)
                    y[-1][-1] += list(dt)




    if False:
        for row in range(16):
            for col in range(16):
                try:
                    p0 = [12, -0.15]
                    tw_corr, var_matrix = curve_fit(
                        linear,
                        x[row][col],
                        y[row][col],
                        p0=p0)
                    tw_corr_0[row][col] = tw_corr[0]
                    tw_corr_1[row][col] = tw_corr[1]
                except ValueError:
                    print("Not enough data to fit")





    # Plotting
    print("Plotting")

    for row in range(16):
        for col in range(16):
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            toa_hist[:,row,col].plot1d(ax=ax)
            fig.savefig(f"{plot_dir}/toa_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
            plt.close()

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    toa_hist.project('toa').plot1d(ax=ax)
    fig.savefig(f"{plot_dir}/toa_total_run_start_{run_start}_run_stop_{run_stop}.png")

    # TOT
    for row in range(16):
        for col in range(16):
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            tot_hist[:,row,col].plot1d(ax=ax)
            fig.savefig(f"{plot_dir}/tot_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
            plt.close()

    for row in range(16):
        for col in range(16):
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            cal_hist[:,row,col].plot1d(ax=ax)
            fig.savefig(f"{plot_dir}/cal_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
            plt.close()
    

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    bin_centers = trigger_hist.axes.centers[0]
    p0=[500,12,5]
    coeff, var_martix = curve_fit(gaus, bin_centers, trigger_hist.values(), p0=p0)
    plt.plot(
        bin_centers,
        gaus(bin_centers, *coeff),
        color='red',
        label='Gaussian Fit\n mean: {:.2f} \n sigma: {:.3f}'.format(coeff[1],abs(coeff[2])),
    )
    trigger_hist.plot1d(ax=ax)
    plt.legend()
    fig.savefig(f"{plot_dir}/trigger_run_start_{run_start}_run_stop_{run_stop}.png")

    # Clock hist
    clock_hist.fill(
        time = events['Clock'],
    )

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    clock_hist.plot1d(ax=ax)
    fig.savefig(f"{plot_dir}/clock_run_start_{run_start}_run_stop_{run_stop}.png")

    ## Delta T 2D profiled distribution and resolution
    #dt_hist_p = dt_hist.profile("time")
    #fig, ax = plt.subplots(1,1,figsize=(10,10))
    #dt_hist_p.plot2d(ax=ax)
    #fig.savefig(f"{plot_dir}/dt.png")

    ## unfortunately, the profile doesn't (properly?) store the width of the distribution
    #fig, ax = plt.subplots(1,1,figsize=(20,20))
    #std_devs = np.sqrt(dt_hist_p.variances())*1000
    #im = ax.matshow(std_devs)
    #cbar = ax.figure.colorbar(im)
    #for i in range(std_devs.shape[0]):
    #    for j in range(std_devs.shape[1]):
    #        c = std_devs[j,i]
    #        ax.text(i, j, "%.0f"%c, va='center', ha='center')
    #ax.set_xlabel('Row')
    #ax.set_ylabel('Col')
    #fig.savefig(f'{plot_dir}/dt_res.png')

    print("Fitting")

    for row in range(16):
        for col in range(16):
            print(f"Timewalk {row=}, {col=}")
            p0 = [12, -0.15] if not args.polynomial else [12, -0.15, 0.1, 0.1]
            tw_corr = p0
            try:
                dt_prof_hist = dt_tot_hist[{'row':row, 'col':col}].profile('time')
                bin_centers = dt_prof_hist.axes.centers[0]
                if not args.unbinned:
                    tw_corr, var_matrix = curve_fit(
                        linear if not args.polynomial else polynomial,
                        bin_centers,
                        dt_prof_hist.values(),
                        check_finite=False,
                        sigma=np.ones_like(dt_prof_hist.values())*0.1,
                        p0=p0)

                    tw_corr_0[row][col] = tw_corr[0]
                    tw_corr_1[row][col] = tw_corr[1]
                    if args.polynomial:
                        tw_corr_2[row][col] = tw_corr[2]
                        tw_corr_3[row][col] = tw_corr[3]

                else:
                    # poor (memory) man's approach to an unbinned fit, just remove the bins with 0 entries
                    try:
                        tw_corr, var_matrix = curve_fit(
                            linear if not args.polynomial else polynomial,
                            bin_centers[dt_prof_hist.values()>0],
                            dt_prof_hist.values()[dt_prof_hist.values()>0],
                            p0=p0
                        )
                        tw_corr_0[row][col] = tw_corr[0]
                        tw_corr_1[row][col] = tw_corr[1]
                        if args.polynomial:
                            tw_corr_2[row][col] = tw_corr[2]
                            tw_corr_3[row][col] = tw_corr[3]
                    except ValueError:
                        print("Fit fails because of only empty bins")
                    except TypeError:
                        print("Fit fails because of too few filled bins")




                fig, ax = plt.subplots(1,1,figsize=(10,10))
                plt.plot(
                    bin_centers,
                    linear(bin_centers, *tw_corr) if not args.polynomial else polynomial(bin_centers, *tw_corr),
                    color='red',
                    label='Linear Fit\n a: {:.2f} \n b: {:.2f}'.format(tw_corr[0],tw_corr[1]),
                )
                #dt_hist[:,row,col].plot1d(ax=ax)
                ax.set_title(f"Row {row} Col {col}")
                ax.set_ylabel("Events")
                ax.set_xlabel(r"ToT")
                ax.set_ylim(10.9,13.1)
                plt.legend()
                # Done second to keep off the legend
                dt_prof_hist.plot1d(ax=ax, label=f"{row=}, {col=}")
                fig.savefig(f"{plot_dir}/timewalk_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
                plt.close()
            except KeyError:
                print(f"Probably no events for {row=}, {col=}")


            print(f"Delta T {row=}, {col=}")
            try:
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                # Do a simple gaussian fit
                bin_centers = dt_hist2[{'row':row, 'col':col}].axes.centers[0]
                p0=[500,12,5]
                try:
                    coeff, var_martix = curve_fit(gaus, bin_centers, dt_hist2[{'row':row, 'col':col}].values(), p0=p0)
                except RuntimeError:
                    coeff = p0
                plt.plot(
                    bin_centers,
                    gaus(bin_centers, *coeff),
                    color='red',
                    label='Gaussian Fit\n mean: {:.2f} \n sigma: {:.3f}'.format(coeff[1],abs(coeff[2])),
                )
                #dt_hist[:,row,col].plot1d(ax=ax)
                ax.set_title(f"Row {row} Col {col}")
                ax.set_ylabel("Events")
                ax.set_xlabel(r"$\Delta t\ (ns)$")
                plt.legend()
                # Done second to keep off the legend
                dt_hist2[{'row':row, 'col':col}].plot1d(ax=ax, label=f"{row=}, {col=}")
                fig.savefig(f"{plot_dir}/dt_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
                plt.close()

                res[row][col] = abs(coeff[2])*1000
            except KeyError:
                print(f"Probably no events for {row=}, {col=}")


    fig, ax = plt.subplots(1,1)
    plt.title("Single Pixel Timing Resolution")
    img = ax.matshow(res)
    fig.colorbar(img, ax = ax)
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(16))

    for i in range(16):
        for j in range(16):
            if res[i, j] > 0:
                text = ax.text(j, i, int(res[i, j]),
                    ha="center", va="center", color="w", fontsize="xx-small")
            else:
                text = ax.text(j, i, '-',
                    ha="center", va="center", color="w", fontsize="xx-small")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    #plt.show()
    fig.savefig(f"{plot_dir}/dt_heatmap_run_start_{run_start}_run_stop_{run_stop}.png")
    plt.close()




    print("Running corrections")

    for events in uproot.iterate(in_files, step_size=step_size):
        events['bin'] = 3.125 / np.maximum(events.cal_code, 1)
        events['toa'] = 12.5 - events.bin * events.toa_code
        events['tot'] = (2*events.tot_code - np.floor(events.tot_code/32))*events.bin
        events['trigger'] = events[lp_val][:,1] * 10**9
        events['dt'] = events['toa'] - (events['trigger'] - events['Clock'])
        for row in range(16):
            for col in range(16):

                scope_sel = ((events.amp[:,1] > 50) & (events.amp[:,1] < 300))
                pix_sel = ((events.row==row)&(events.col==col))
                #cal_hist2.fill(
                #    cal=ak.flatten(events.cal_code[pix_sel])
                #)
                #cal_peak = np.where(cal_hist2.values()==np.max(cal_hist2.values()))[0][0]
                cal_peak = cal_mean[row][col]
                #print(row, col, cal_peak)
                cal_sel = ((events.cal_code<(cal_peak+2)) & (events.cal_code>(cal_peak-2)))
                toa_sel = ((events.toa_code>100) & (events.toa_code<500))
                tot_sel = (events.tot_code>90)
                sel = pix_sel & cal_sel & (scope_sel) & toa_sel & tot_sel

                #cal_mean = ak.mean(events.cal_code[sel])
                #cal = events.cal_code
                time_bin = 3.125 / cal_mean[row][col]
                toa = 12.5 - time_bin * events.toa_code
                tot = ((2*events.tot_code - np.floor(events.tot_code/32))*time_bin)[sel]
                dt = (toa - (events.trigger - events.Clock))[sel]

                if args.polynomial:
                    dt_corr = ak.flatten(dt)-(tw_corr_0[row][col]+tw_corr_1[row][col]*ak.flatten(tot)+tw_corr_2[row][col]*ak.flatten(tot)**2 + tw_corr_3[row][col]*ak.flatten(tot)**3)
                else:
                    dt_corr = ak.flatten(dt)-(tw_corr_0[row][col]+tw_corr_1[row][col]*ak.flatten(tot))

                dt_hist3.fill(
                    row=row,
                    col=col,
                    time=dt_corr,
                )




    print("Plotting corrected delta T")

    #import ROOT
    #timeRes2D = ROOT.TH2D("","",16,0,16, 16,0,16)
    res = np.zeros([16, 16])
    var = np.zeros([16, 16])
    for row in range(16):
        for col in range(16):
            try:
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                # Do a simple gaussian fit
                bin_centers = dt_hist3[{'row':row, 'col':col}].axes.centers[0]
                p0=[500,0,5]
                try:
                    coeff, var_matrix = curve_fit(gaus, bin_centers, dt_hist3[{'row':row, 'col':col}].values(), p0=p0)
                except RuntimeError:
                    coeff = p0
                    var_matrix = np.zeros([3,3])
                plt.plot(
                    bin_centers,
                    gaus(bin_centers, *coeff),
                    color='red',
                    label=r'Fit mean: {:.2f}, $\sigma: {:.0f} \pm {:.1f}$ (ps)'.format(coeff[1],abs(coeff[2])*1000, np.sqrt(var_matrix[2,2])*1000)
                    )

                #timeRes2D.SetBinContent(row, col, coeff[1])
                #timeRes2D.SetBinError(row, col, abs(coeff[2])*1000)

                #dt_hist[:,row,col].plot1d(ax=ax)
                ax.set_title(f"Row {row} Col {col}")
                ax.set_ylabel("Events")
                ax.set_xlabel(r"$\Delta t\ (ns)$")
                plt.legend()
                # Done second to keep off the legend
                dt_hist3[{'row':row, 'col':col}].plot1d(ax=ax, label=f"{row=}, {col=}")
                fig.savefig(f"{plot_dir}/dt_corrected_row_{row}_col_{col}_run_start_{run_start}_run_stop_{run_stop}.png")
                plt.close()
                if abs(coeff[2])*1000 > 2000:
                    print(f"Probably no events for {row=}, {col=}")
                    res[row, col] = -1
                else:
                    res[row, col] = abs(coeff[2])*1000
                    var[row, col] = var_matrix[2,2]
            except KeyError:
                print(f"Probably no events for {row=}, {col=}")
                res[row, col] = -1

    fig, ax = plt.subplots(1,1)
    plt.title("Single Pixel Timing Resolution")
    img = ax.matshow(res)
    fig.colorbar(img, ax = ax)
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(16))
    
    for i in range(16):
        for j in range(16):
            if res[i, j] > 0:
                text = ax.text(j, i, int(res[i, j]),
                    ha="center", va="center", color="w", fontsize="xx-small")
            else:
                text = ax.text(j, i, '-',
                    ha="center", va="center", color="w", fontsize="xx-small")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row") 
    #plt.show()
    fig.savefig(f"{plot_dir}/dt_corrected_heatmap_run_start_{run_start}_run_stop_{run_stop}.png")
    plt.close()
    
    #timeRes2D.Draw("colz text")
