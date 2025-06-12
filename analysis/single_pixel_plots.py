import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import pickle
import json
import os
from tqdm import tqdm
import argparse
from scipy import stats
import sys
import yaml

def loadData(path):
    fileform = f'Qinj_scan_'
    files = [f for f in os.listdir(path) if fileform in f]

    #Loading In Data
    if len(files) == 0:
        print('No files here.')
        sys.exit(0)

    print(f'Loading the following files from {path}')

    df = pd.DataFrame()
    if any(['.json' in f for f in files]):
        files = [f for f in files if '.json' in f]
    elif any(['.pkl' in f for f in files]):
        files = [f for f in files if '.pkl' in f]

    for f in files:
        print(f)
        charge = int(f.split('_')[-1].split('.')[0])
        if '.pkl' in f:
            sub = pickle.load(open(path+f, 'rb'))
            outfile = f.replace('.pkl', '.json')
            with open(path + outfile, 'w') as out:
                json.dump(sub.to_dict(), out)
        elif '.yaml' in f:
            with open(path + f, 'r') as infile:
                sub = pd.DataFrame(yaml.load(infile, Loader = yaml.FullLoader))
            outfile = f.replace('.yaml', '.json')
            with open(path + outfile, 'w') as out:
                json.dump(sub.to_dict(), out)
        else:
            with open(path + f, 'r') as infile:
                sub = pd.DataFrame(json.load(infile))
        sub['charge'] = [charge]*len(sub.vth)
        if True:#not len(np.unique(sub.hits)) == 1:
            df = pd.concat([df, sub])
    return df

def makeDFCuts(df, args):
    new = pd.DataFrame(columns = df.columns)
    for q in np.unique(df.charge):
        sub = df[df.charge == q]
        idx = True

        idx = idx&(sub.hits > args.hits_cut*args.nl1a)
        idx = idx&(sub.vth > args.vth_low)
        idx = idx&(sub.vth < args.vth_high)
        new = pd.concat([new, sub[idx]])
    return new

def makeCodeCuts(df, args):
    new = pd.DataFrame(columns = df.columns.tolist() + ['toacode', 'totcode'])
    n = 0
    for q in np.unique(df.charge):
        sub = df[df.charge == q]
        for i in range(len(sub)):
            idx = True
            totcode = np.array(sub.tot.iloc[i])
            toacode = np.array(sub.toa.iloc[i])
            cal = np.array(sub.cal.iloc[i])
            u, c = np.unique(cal, return_counts = True)
            calmode = u[np.argmax(c)]
            idx = idx&(np.abs(cal - calmode) < 2)
            idx = idx&(totcode < args.tot_high)
            idx = idx&(totcode > args.tot_low)
            idx = idx&(toacode < args.toa_high)
            idx = idx&(toacode > args.toa_low)
            if np.sum(idx) > 10:
                tbin = 3.125/cal[idx]
                toa = tbin*toacode[idx]
                tot = (2*totcode[idx] - totcode[idx]//32)/tbin
                row = {
                    'totcode':totcode[idx].tolist(),
                    'toacode':toacode[idx].tolist(),
                    'cal':cal[idx].tolist(),
                    'tot':tot.tolist(),
                    'toa':toa.tolist(),
                    'hits':len(tot),
                    'vth':sub.vth.iloc[i],
                    'charge':q
                    }
                new.loc[n] = row
                n += 1
    return new

def plotHitsvDAC(df, pa, args, mode):
    HitsvDACplotter(df, pa, args, mode=mode)
    for q in np.unique(df.charge):
        HitsvDACplotter(df, pa, args, charge = q, mode=mode)

def HitsvDACplotter(df, pa, args, charge = None, mode = 'pre'):
    fig = plt.figure(figsize = pa['figsize'])
    plt.title(f'Hits v. Threshold DAC for {pa["loc_title"]}\nDelay = {args.delay}, # of L1A triggers: {args.nl1a}', fontsize = pa['titfontsize'])
    plt.xlabel('Threshold DAC Values', fontsize = pa['labfontsize'])
    plt.ylabel('Hits',  fontsize = pa['labfontsize'])
    if not charge:
        for q in tqdm(np.unique(df.charge)):
            idx = df.charge == q
            idx = idx&(df.vth > args.vth_low)
            idx = idx&(df.vth < args.vth_high)
            x = df.vth[idx]
            y = df.hits[idx]
            plt.plot(x, y, 'o-', label = f'Qinj = {q}')
        plt.legend()
        if mode == 'pre': 
            savename = f'{pa["store"]}/DAC_v_Hits.pdf'
        else:
            savename = f'{pa["store"]}/DAC_v_Hits_redux.pdf'
    else:
        idx = df.charge == charge
        x = df.vth[idx]
        y = df.hits[idx]
        plt.plot(x, y, 'o-')
        if mode == 'pre': 
            savename = f'{pa["store"]}/DAC_v_Hits_q_{charge}.pdf'
        else:
            savename = f'{pa["store"]}/DAC_v_Hits_q_{charge}_redux.pdf'
    plt.legend()
    plt.savefig(savename)
    plt.savefig(savename.replace('pdf', 'png'))
    if args.show_plots:
        plt.show()
    plt.close()


def plotFallingEdge(df, pa, args):
    fig, ax = plt.subplots(figsize = pa['figsize'])
    charges = []
    firstbest = []
    hitslim = args.hits_cut*args.nl1a
    for q in np.unique(df.charge):
        data = df[df.charge == q]
        best = np.max(data.hits)
        if best >= hitslim:
            charges.append(q)
            firstbest.append(data.vth[data.hits >= hitslim].iloc[-1])
    ax.scatter(charges, firstbest, label = 'Data', color = 'r')
    model = stats.linregress(charges, firstbest)
    x = np.linspace(np.min(charges), np.max(charges), 1000)
    y = x*model.slope + model.intercept
    ax.plot(x, y, label = 'Fit', color = 'b')

    ax.set_title('Start of S-Curve Falling Edge')
    ax.set_xlabel('Injected Charge')
    ax.set_ylabel('Threshold DAC')
    plt.savefig(f'{pa["store"]}/Threshold_DAC_Limit.png')
    plt.savefig(f'{pa["store"]}/Threshold_DAC_Limit.png')
    if args.show_plots:
        plt.show()
    plt.close()

def plotSingleVthDists(df, pa, args):
    print('Working on TOT, TOA, and Cal hists for individual settings')
    for q in tqdm(np.unique(df.charge)):
        for d in tqdm(args.plotted_vths, leave = False, desc = f'Working on QSel = {q}'):
            #if not d in df.vth: d = np.random.choice(df.vth, 1)
            idx = (df.charge == q)&(df.vth == d)
            if np.sum(idx) == 0:
                continue
            for att in ['tot', 'cal', 'toa']:
                fig = plt.figure(figsize = pa['figsize'])
                data = df[att][idx].iloc[0]
                plt.hist(data, bins = args.nbins, density = True)#bins = range(np.min(tot), np.max(tot) + 1), density = True)
                plt.title(f'{att.upper()} Codes for Threshold DAC = {d} {pa["loc_title"]}\nEntries: {args.nl1a}, Delay = {args.delay}, Qinj = {q}')
                plt.xlabel(f'{att.upper()} Values')
                plt.ylabel('Frequency')
                plt.yscale('log')
                plt.savefig(f'{pa["store"]}/{att.upper()}_vth_{d}_q{q}.pdf')
                plt.savefig(f'{pa["store"]}/{att.upper()}_vth_{d}_q{q}.png')
                plt.close()


def plotCodesvDAC(df, pa, args, code):
    print('Working on TOA v. DAC')

    if code:
        mode = 'Code'
    else:
        mode = 'Value'
    data = {}
    atts = ['tot', 'toa', 'cal']
    for q in tqdm(np.unique(df.charge)):
        idx = df.charge == q
        data[q] = {}

        data[q]['vth'] = df.vth[idx]

        toa = df.toa[idx]
        data[q]['toaavg'] = [np.mean(dat) for dat in toa]
        data[q]['toastd'] = [np.std(dat) for dat in toa]

        tot = df.tot[idx]
        data[q]['totavg'] = [np.mean(dat) for dat in tot]
        data[q]['totstd'] = [np.std(dat) for dat in tot]

        cal = df.cal[idx]
        data[q]['calavg'] = [np.mean(dat) for dat in cal]
        data[q]['calstd'] = [np.std(dat) for dat in cal]
        
        for att in atts:
            x = data[q]['vth']
            y = data[q][att + 'avg']
            yerr = data[q][att + 'std']
            ymax = np.max(np.array(y)[np.isfinite(y)])
            ymin = np.min(np.array(y)[np.isfinite(y)])
            fig = plt.figure(figsize = pa['errorbarsize'])
            plt.errorbar(x, y, yerr, fmt = 'o-', capsize = 3)
            plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
            plt.ylabel(f'{att.upper()} {mode} Mean', fontsize = pa['labfontsize'])
            plt.title(f'Mean {att.upper()} {mode} vs. Theshold DAC for Delay = {args.delay} {pa["loc_title"]}, Qinj = {q}', fontsize = pa['titfontsize'])
            bar = 0.15*ymax
            #plt.ylim([ymin - bar, ymax + bar])
            plt.savefig(f'{pa["store"]}/DAC_v_{att.upper()}{mode}_q{q}.pdf')
            plt.savefig(f'{pa["store"]}/DAC_v_{att.upper()}{mode}_q{q}.png')
            plt.close()

    for att in atts:
        fig = plt.figure(figsize = pa['errorbarsize'])
        ymax = 0
        ymin = 1e10
        for q in tqdm(np.unique(df.charge)):
            x = data[q]['vth']
            y = data[q][att + 'avg']
            yerr = data[q][att + 'std']
            ymaxt = np.max(np.array(y)[np.isfinite(y)])
            ymax = np.max([ymaxt, ymax])
            ymint = np.min(np.array(y)[np.isfinite(y)])
            ymin = np.min([ymint, ymin])
            plt.errorbar(data[q]['vth'], data[q][att + 'avg'], data[q][att + 'std'], fmt = 'o-', capsize = 3, label = f'Qinj = {q}')
        plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
        plt.ylabel(f'{att.upper()} {mode} Mean', fontsize = pa['labfontsize'])
        plt.title(f'Mean {att.upper()} vs. Theshold DAC for Delay = {args.delay} {pa["loc_title"]}', fontsize = pa['titfontsize'])
        plt.legend()
        bar = 0.15*ymax
        #plt.ylim([ymin - bar, ymax + bar])
        plt.savefig(f'{pa["store"]}/DAC_v_{att.upper()}{mode}.pdf')
        plt.savefig(f'{pa["store"]}/DAC_v_{att.upper()}{mode}.png')
        if args.show_plots:
            plt.show()
        plt.close()

def plotTOASDvDAC(df, pa, args):
    data = {}
    atts = ['tot', 'toa', 'cal']
    for q in tqdm(np.unique(df.charge)):
        idx = df.charge == q
        data[q] = {}
        data[q]['vth'] = df.vth[idx].tolist()
        data[q]['toastd'] = [np.std(dat) for dat in df[idx].toa]
        data[q]['toacodestd'] = [np.std(dat) for dat in df[idx].toacode]
        for dat in df[idx].toacode:
            print(dat)
        fig = plt.figure(figsize = pa['errorbarsize'])
        plt.plot(data[q]['vth'], data[q]['toastd'], 'o-')
        plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
        plt.ylabel('TOA Standard Deviation', fontsize = pa['labfontsize'])
        plt.title(f'TOA Standard Deviation vs. Theshold DAC\n for Delay = {args.delay} {pa["loc_title"]}, Qinj = {q}', fontsize = pa['titfontsize'])
        plt.savefig(f'{pa["store"]}/DAC_v_TOASD_q{q}.pdf')
        plt.savefig(f'{pa["store"]}/DAC_v_TOASD_q{q}.png')
        plt.close()

        fig = plt.figure(figsize = pa['errorbarsize'])
        plt.plot(data[q]['vth'], data[q]['toacodestd'], 'o-')
        plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
        plt.ylabel('TOA Code Standard Deviation', fontsize = pa['labfontsize'])
        plt.title(f'TOA Code Standard Deviation vs. Theshold DAC\n for Delay = {args.delay} {pa["loc_title"]}, Qinj = {q}', fontsize = pa['titfontsize'])
        plt.savefig(f'{pa["store"]}/DAC_v_TOACODESD_q{q}.pdf')
        plt.savefig(f'{pa["store"]}/DAC_v_TOACODESD_q{q}.png')
        plt.close()


    fig = plt.figure(figsize = pa['errorbarsize'])
    for q in tqdm(np.unique(df.charge)):
        plt.plot(data[q]['vth'], data[q]['toastd'], 'o-', label = f'Qinj = {q}')
    plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
    plt.ylabel('TOA Standard Deviation', fontsize = pa['labfontsize'])
    plt.title(f'TOA Standard Deviation vs. Theshold DAC\n for Delay = {args.delay} {pa["loc_title"]}', fontsize = pa['titfontsize'])
    plt.legend()
    plt.savefig(f'{pa["store"]}/DAC_v_TOASD.pdf')
    plt.savefig(f'{pa["store"]}/DAC_v_TOASD.png')
    if args.show_plots:
        plt.show()
    plt.close()

    fig = plt.figure(figsize = pa['errorbarsize'])
    for q in tqdm(np.unique(df.charge)):
        plt.plot(data[q]['vth'], data[q]['toacodestd'], 'o-', label = f'Qinj = {q}')
    plt.xlabel('Threshold DAC', fontsize = pa['labfontsize'])
    plt.ylabel('TOA Code Standard Deviation', fontsize = pa['labfontsize'])
    plt.title(f'TOA Code Standard Deviation vs. Theshold DAC\n for Delay = {args.delay} {pa["loc_title"]}, Qinj = {q}', fontsize = pa['titfontsize'])
    plt.legend()
    plt.savefig(f'{pa["store"]}/DAC_v_TOACODESD.pdf')
    plt.savefig(f'{pa["store"]}/DAC_v_TOACODESD.png')
    if args.show_plots:
        plt.show()
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--loc', action = 'store', default = '4,3')
parser.add_argument('--input', action = 'store')

parser.add_argument('--vth_high', action = 'store', type = int, default = 1000)
parser.add_argument('--vth_low', action = 'store', type = int, default = 0)
parser.add_argument('--cal_cuts', action = 'store', nargs = 2, type = int, default = [5000])
parser.add_argument('--tot_low', action = 'store', type = int, default = 0)
parser.add_argument('--tot_high', action = 'store', type = int, default = 1000)
parser.add_argument('--toa_low', action = 'store', type = int, default = 0)
parser.add_argument('--toa_high', action = 'store', type = int, default = 1000)

parser.add_argument('--nl1a', action = 'store', type = int, default = 3200)
parser.add_argument('--delay', action = 'store', type = int, default = 504)
parser.add_argument('--nbins', action = 'store', type = int, default = 10)
parser.add_argument('--plotted_vths', action = 'store', nargs = '*', default = [340], type = int)
parser.add_argument('--hits_cut', action = 'store', type = float, default = 0.9)
parser.add_argument('--etroc', action = 'store', default = '0')
parser.add_argument('--show_plots', action = 'store_true')
args = parser.parse_args()

if args.loc == 'broadcast':
    loc_title = 'Broadcasted'
    pix_path = 'broadcast/'
else:
    i = int(args.loc.split(',')[0])
    j = int(args.loc.split(',')[1])
    loc_title = f'ETROC {args.etroc}, Row {i} Col {j}'
    pix_path = f'/r{i}c{j}/'

pa = {
        'labfontsize':20,
        'titfontsize':25,
        'store':args.input + 'qinj_plots/',
        'figsize':(9, 7),
        'errorbarsize':(12, 7),
        'loc_title':loc_title
        }


if not os.path.isdir(pa['store']):
    os.mkdir(pa['store'])

df = loadData(args.input)

print()
print(df.head(30))
if df.empty:
    print('Files found but no hits found.')
    sys.exit(0)

#Slide 3 S Curve Plots
#Hits v. Threshold DAC Values, All QSel

plotHitsvDAC(df, pa, args, mode = 'pre')

df = makeDFCuts(df, args)

plotHitsvDAC(df, pa, args, mode = 'redux')

plotFallingEdge(df, pa, args)

plotSingleVthDists(df, pa, args)

plotCodesvDAC(df, pa, args, code = True)

df = makeCodeCuts(df, args)
print(df.head(30))
plotCodesvDAC(df, pa, args, code = False)

plotTOASDvDAC(df, pa, args)

'''




# TOA v. TOT 

u, c = np.unique(ak.flatten(df.cal), return_counts = True)
calcut = u[np.argmax(c)]
toalim = 8
totlim = 8
callim = 0.5
for q in np.unique(df.charge):
    x = []
    y = []
    chargeidx = df.charge== q
    i = df[(chargeidx)].hits.argmax()
    n = int(np.min([30, len(df[(chargeidx)].hits) - i - 1]))
    while df[(chargeidx)].hits.iloc[i] > df[(chargeidx)].hits.iloc[i+n]:
        n = n//2
    i+=n
    x = np.array(df.toa.iloc[i])
    y = np.array(df.tot.iloc[i])
    cal = np.array(df.cal.iloc[i])
    caldiff = np.abs(cal - calcut) < callim
    toadiff = np.abs(x - np.mean(x)) < toalim*np.std(x)
    totdiff = np.abs(y - np.mean(y)) < totlim*np.std(y)
    idx = caldiff*totdiff*toadiff
    x = x[idx]
    y = y[idx]
    cal = cal[idx]
    tbin = 3.125/cal
    x = tbin*x
    y = tbin*(2*y-y//32)
    fig, ax = plt.subplots()
    h = ax.hist2d(x, y, bins = [50, 50])#, norm = 'log')
    fig.colorbar(h[3], ax = ax)
    ax.set_title(f'TOT v. TOA {loc_title}\nHits = {len(x)}/{df.hits.iloc[i]}, Qinj = {q}\nDAC = {df.vth.iloc[i]}')
    ax.set_xlabel('Mean TOT')
    ax.set_ylabel('Mean TOA')
    plt.show()
    plt.savefig(f'{store}/TOT_v_TOA_single_q{q}.png')
    plt.savefig(f'{store}/TOT_v_TOA_single_q{q}.pdf')
    plt.close()



for q in np.unique(df.charge):
    x = []
    y = []
    cal = []
    chargeidx = df.charge== q
    i = df[(chargeidx)].hits.argmax()
    for i in range(len(df[(chargeidx)].hits)):
        x += df[(chargeidx)].toa.iloc[i]
        y += df[(chargeidx)].tot.iloc[i]
        cal += df[(chargeidx)].cal.iloc[i]
    x = np.array(x)
    y = np.array(y)
    cal = np.array(cal)
    caldiff = np.abs(cal - calcut) < callim
    toadiff = np.abs(x - np.mean(x)) < toalim*np.std(x)
    totdiff = np.abs(y - np.mean(y)) < totlim*np.std(y)
    idx = caldiff*totdiff*toadiff
    x = x[idx]
    y = y[idx]
    cal = cal[idx]
    tbin = 3.125/cal
    x = tbin*x
    y = tbin*(2*y-y//32)
    fig, ax = plt.subplots()
    h = ax.hist2d(x, y, bins = [50, 50])#, norm = 'log')
    fig.colorbar(h[3], ax = ax)
    ax.set_title(f'TOT v. TOA {loc_title}\nHits = {len(x)}/{df.hits.iloc[i]}, Qinj = {q}\nDAC = {df.vth.iloc[i]}')
    ax.set_xlabel('Mean TOT')
    ax.set_ylabel('Mean TOA')
    plt.show()
    plt.savefig(f'{store}/TOT_v_TOA_q{q}.png')
    plt.savefig(f'{store}/TOT_v_TOA_q{q}.pdf')
    plt.close()

'''
