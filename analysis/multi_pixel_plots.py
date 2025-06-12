import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import pickle
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', action = 'store')
parser.add_argument('--charge', '-q', action = 'store', type = int)
parser.add_argument('--vth_axis', action = 'store', default = [], nargs = 2, type = int)
parser.add_argument('--count_lim', action = 'store', default = 0, type = float)
parser.add_argument('--cal_lim', action = 'store', default = 500000, type = int)
parser.add_argument('--nl1a', action = 'store', type = int, default = 32000)
args = parser.parse_args()

q = args.charge
if len(args.vth_axis) != 2:
    if q == 25:
        exrange = [220, 550]
    else:
        exrange = [220, 425]
else:
    exrange = args.vth_axis
nl1a = args.nl1a
countlim = args.count_lim*nl1a
callim = args.cal_lim
path = args.input
pixels = [f for f in os.listdir(path) if not '.' in f]#[:15]
print(pixels)

df = pd.DataFrame()
temp = []
for pix in pixels:
    i = int(pix.split('c')[0][1:])
    j = int(pix.split('c')[1])
    pixpath = path + pix + f'/Qinj_scan_L1A_504_{q}.pkl'
    print(pixpath)
    if os.path.exists(pixpath):
        try:
            data = pickle.load(open(pixpath, 'rb'))
            if len(np.unique(data.hits)) > 1:
                data['row'] = [i]*len(data)
                data['col'] = [j]*len(data)
                df = pd.concat([df, data])
                print(f'Successfully pulled data for Row {i} Col {j}.')
                temp.append(pix)
        except:
            print(f'Found but failed to retrieve data for Row {i} Col {j}.')
    else:
        print(f'Found folder but no files for Row {i} Col {j}.')
pixels = temp
print(df)
print(df[df.hits > 0].head())

fitdata = {}
fig, ax = plt.subplots(figsize = (9,7))
if len(pixels) <= 5:
    subset = pixels
else:
    #subset = np.random.choice(pixels, 5, replace = False)
    subset = ['r2c5', 'r3c8', 'r13c11', 'r15c1', 'r8c9']
calgrid = np.zeros([16, 16])
calstdgrid = np.zeros([16, 16])
calgridtxt = [[i for i in range(16)] for i in range(16)]
calstdgridtxt = [[i for i in range(16)] for i in range(16)]
for i in range(16):
    for j in range(16):
        data = df[(df.row == i)&(df.col == j)]
        cal = []
        for n in range(len(data)):
            if data.hits.iloc[n] == nl1a:
                cal += data.cal.iloc[n]
        print(i, j, len(cal))
        if len(cal) == 0:
            calgrid[i, j] = 5000
            calstdgrid[i, j] = 5000
            calgridtxt[j][i] = '-'
            calstdgridtxt[j][i] = '-'
        else:
            cals = np.unique(cal)
            calcounts = [cal.count(c) for c in cals]
            calgrid[i, j] = cals[np.argsort(-np.array(calcounts))[0]]
            #calgrid[i, j] = np.mean(cal)
            calstdgrid[i, j] = np.std(cal)
            calgridtxt[j][i] = str(int(np.mean(cal)))
            calstdgridtxt[j][i] = str(np.round(np.std(cal), 2))

for i in range(16):
    for j in range(16):
        if calgridtxt[i][j] == '-':
            calgrid[j, i] = np.min(calgrid)
            calstdgrid[j, i] = np.min(calstdgrid)

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'CAL Code Means Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(calgrid) - np.min(calgrid)) + np.min(calgrid)
im = ax.imshow(calgrid, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if calgrid[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, calgridtxt[i][j], fontsize = 8, ha = 'center', va = 'center', color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_CAL_{q}.pdf')
plt.savefig(path + f'DAC_v_CAL_{q}.png')

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'CAL Code STD Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(calstdgrid) - np.min(calstdgrid)) + np.min(calstdgrid)
im = ax.imshow(calstdgrid, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if calstdgrid[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, calstdgridtxt[i][j], fontsize = 8, ha = 'center', va = 'center', color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_CALSTD_{q}.pdf')
plt.savefig(path + f'DAC_v_CALSTD_{q}.png')
plt.close()

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'Time of Arrival Standard Deviation v. Threshold DAC\nQinj = {q}')
ax.set_xlabel('Threshold DAC')
ax.set_ylabel('TOA Std. Dev.')
for pix in subset:
    i = int(pix.split('c')[0][1:])
    j = int(pix.split('c')[1])
    data = df[(df.row == i)&(df.col == j)]
    
    x = []
    y = []
    for n in range(len(data)):
        subtoa = np.array(data.toa.iloc[n])
        subcal = np.array(data.cal.iloc[n])
        if len(subtoa) > countlim:
            subtoa = subtoa[(np.abs(subcal - calgrid[i, j]) < callim).nonzero()[0]]
            if len(subtoa) > countlim and np.std(subtoa) < 2:
                x.append(data.vth.iloc[n])
                y.append(np.std(subtoa))
    if len(x) > 10:
        transformer = PolynomialFeatures(degree = 4)
        model = LinearRegression()
        X = transformer.fit_transform(np.array(x).reshape(-1, 1))
        model.fit(X, y)
        X = transformer.fit_transform(np.array(data.vth).reshape(-1, 1))
        fitdata[pix] = {'x': data.vth, 'y':model.predict(X), 'ydat':y}
        ax.plot(x, y, 'o', label = f'Row {i} Col {j}')


plt.plot(exrange, [0.5, 0.5], label = 'TOA SD = 0.5')
plt.plot(exrange, [1, 1], label = 'TOA SD = 1')

ax.set_ylim([0, 2])
ax.set_xlim(exrange)
plt.legend()
plt.savefig(path + f'DAC_v_TOASD_multi_{q}.png')
plt.savefig(path + f'DAC_v_TOASD_multi_{q}.pdf')
plt.close()

plt.figure(figsize = (9,7))
plt.title(f'Hits v. Threshold DAC\nQinj = {q}')
plt.xlabel('Threshold DAC')
plt.ylabel('Hits')

fig, ax = plt.subplots(figsize = (9,7))

ax.set_title(f'Time of Arrival Standard Deviation Fits v. Threshold DAC\nQinj = {q}')
ax.set_xlabel('Threshold DAC')
ax.set_ylabel('TOA Std. Dev. Prediction')

for pix in fitdata:
    i = pix.split('c')[0][1:]
    j = pix.split('c')[1]
    ax.plot(fitdata[pix]['x'], fitdata[pix]['y'], label = f'Row {i} Col {j}')
ax.set_ylim([0, 2])
ax.set_xlim(exrange)
plt.plot(exrange, [0.5, 0.5], label = 'TOA SD = 0.5')
plt.plot(exrange, [1, 1], label = 'TOA SD = 1')

plt.legend()
plt.savefig(path + f'DAC_v_TOTSD_Fit_multi_{q}.png')

fig, ax = plt.subplots(figsize = (9,7))

ax.set_title(f'CAL Code v. Threshold DAC\nQinj = {q}')
ax.set_xlabel('Threshold DAC')
ax.set_ylabel('TOA Std. Dev. Prediction')

for pix in fitdata:
    i = int(pix.split('c')[0][1:])
    j = int(pix.split('c')[1])
    data = df[(df.row == i)&(df.col == j)]
    x = []
    y = []
    err = []
    for n in range(len(data)):
        if data.hits.iloc[n] > 10:
            x.append(data.vth.iloc[n])
            y.append(np.mean(data.cal.iloc[n]))
            err.append(np.std(data.cal.iloc[n]))
    ax.errorbar(x, y, err, capsize = 5, alpha = 0.7, label = f'Row {i} Col {j}')
#ax.set_ylim([0])
ax.set_xlim(exrange)
#ax2 = ax.secondary_yaxis('right', functions = (forward, backward))
#ax2.set_ylabel('Jitter (ps)')
#plt.plot(exrange, [0.5, 0.5], label = 'TOA SD = 0.5')
#plt.plot(exrange, [1, 1], label = 'TOA SD = 1')

plt.legend()
plt.savefig(path + f'DAC_v_CAL_multi_{q}.png')
plt.savefig(path + f'DAC_v_CAL_multi_{q}.pdf')

mins = np.zeros([16, 16]) - 50000
minstxt = [['u' for i in range(16)] for i in range(16)]
minsconv = np.zeros([16, 16]) - 50000
minsconvtxt = [['u' for i in range(16)] for i in range(16)]
widths = np.zeros([16, 16]) + 5000
widthstxt = [['u' for i in range(16)] for i in range(16)]
datmin = np.zeros([16, 16]) - 50000
dattxt = [['u' for i in range(16)] for i in range(16)]

for i in range(16):
    for j in range(16):
        data = df[(df.row == i)&(df.col == j)]
        pix = f'r{i}c{j}'
        if pix in pixels and not pix in fitdata:
            x = []
            ydat = []
            for n in range(len(data)):
                subtoa = np.array(data.toa.iloc[n])
                subcal = np.array(data.cal.iloc[n])
                if len(subtoa) > countlim:
                    subtoa = subtoa[(np.abs(subcal - calgrid[i, j]) < callim).nonzero()[0]]
                    if len(subtoa) > countlim and np.std(subtoa) < 2:
                        x.append(data.vth.iloc[n])
                        ydat.append(np.std(subtoa))
            if len(x) > 10:
                transformer = PolynomialFeatures(degree = 4)
                model = LinearRegression()
                X = transformer.fit_transform(np.array(x).reshape(-1, 1))
                model.fit(X, ydat)
                X = transformer.fit_transform(np.array(data.vth).reshape(-1, 1))
                y = model.predict(X) 

        elif pix in pixels and pix in fitdata:
            y = fitdata[pix]['y']
            ydat = fitdata[pix]['ydat']

        if pix in pixels:
            try:
                ydiff = [y[n] - y[n-1] for n in range(1, len(y))]
                lim = 0.05
                starter = [diff > -lim for diff in ydiff]
                ender = [diff < lim for diff in ydiff]
                startidx = starter.index(True)
                ender.reverse()
                endidx = len(ender) - ender.index(True)
                if np.min(y) > 2 or np.min(y) < 0:
                    stupidshittodosomenign
                mins[i, j] = np.min(y)
                minstxt[j][i] = f'{np.round(mins[i, j], 2)}'
                widths[i, j] = int(data.vth.iloc[endidx] - data.vth.iloc[startidx])
                widthstxt[j][i] = f'{int(widths[i, j])}'
                datmin[i, j] = np.min(ydat)
                dattxt[j][i] = f'{np.round(datmin[i, j], 2)}'
                print(mins[i, j], minstxt[j][i], widths[i, j], widthstxt[j][i])
            except:
                minstxt[j][i] = 'F'
                widthstxt[j][i] = 'F'
                dattxt[j][i] = 'F'

        else:
            minstxt[j][i] = 'M'
            widthstxt[j][i] = 'M'
            dattxt[j][i] = 'M'

minmin = np.max(mins)
widthmin = np.min(widths)
datminmin = np.max(datmin)
for i in range(16):
    for j in range(16):
        if minstxt[j][i] == 'M' or  minstxt[j][i] == 'F':
            mins[i, j] = minmin
            widths[i,j] = widthmin
            datmin[i, j] = datminmin

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'TOA Std Dev Fit Minima Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
im = ax.imshow(mins, cmap = 'RdYlGn_r')#, cmap = 'Purples')
clim = .6*(np.max(mins) - np.min(mins)) + np.min(mins)
for i in range(16):
    for j in range(16):
        if mins[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        print(mins[i, j], clim, color)        
        ax.text(i, j, minstxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TOASD_Fit_Minima_{q}.pdf')
plt.savefig(path + f'DAC_v_TOASD_Fit_Minima_{q}.png')

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'TOA Std Dev Fit Widths Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(widths) - np.min(widths)) + np.min(widths)
im = ax.imshow(widths, cmap = 'RdYlGn')#, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if widths[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, widthstxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TOASD_Fit_Widths_{q}.pdf')
plt.savefig(path + f'DAC_v_TOASD_Fit_Widths_{q}.png')

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'TOA Std Dev Data Minima Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(widths) - np.min(widths)) + np.min(widths)
im = ax.imshow(datmin, cmap = 'RdYlGn_r')#, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if widths[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, dattxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TOASD_Data_Minima_{q}.pdf')
plt.savefig(path + f'DAC_v_TOASD_Data_Minima_{q}.png')

tmins = mins*0x3/calgrid*1000
tminstxt = [['-' if minstxt[i][j] == '-' else str(np.round(tmins[j, i], 2)) for i in range(16)] for j in range(16)]
twidths = widths*0x3/calgrid*1000
twidthstxt = [['-' if widthstxt[i][j] == '-' else str(np.round(twidths[j, i], 2)) for i in range(16)] for j in range(16)]
tdatmin = datmin*0x3/calgrid*1000
tdattxt = [['-' if dattxt[i][j] == '-' else str(np.round(tdatmin[j, i], 2)) for i in range(16)] for j in range(16)]

fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'True TOA Std Dev Data Minima Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(tdatmin) - np.min(tdatmin)) + np.min(tdatmin)
im = ax.imshow(tdatmin, cmap = 'RdYlGn')#, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if widths[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, tdattxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TTOASD_Data_Minima_{q}.pdf')
plt.savefig(path + f'DAC_v_TTOASD_Data_Minima_{q}.png')


fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'True TOA Std Dev Minima Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(tmins) - np.min(tmins)) + np.min(tmins)
im = ax.imshow(twidths, cmap = 'RdYlGn')#, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if widths[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, tminstxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TTOASD_Fit_Minima_{q}.pdf')
plt.savefig(path + f'DAC_v_TTOASD_Fit_Minima_{q}.png')


fig, ax = plt.subplots(figsize = (9, 7))
ax.set_title(f'True TOA Std Dev Fit Widths Qinj = {q}')
ax.set_xlabel('Col')
ax.set_ylabel('Row')
clim = .6*(np.max(twidths) - np.min(twidths)) + np.min(twidths)
im = ax.imshow(twidths, cmap = 'RdYlGn')#, cmap = 'Purples')
for i in range(16):
    for j in range(16):
        if widths[j, i] > clim:
            color = 'white'
        else:
            color = 'black'
        ax.text(i, j, twidthstxt[i][j], fontsize = 8, ha = 'center', va = 'center')#, color = color)
cbar = ax.figure.colorbar(im, ax = ax)
plt.savefig(path + f'DAC_v_TTOASD_Fit_Widths_{q}.pdf')
plt.savefig(path + f'DAC_v_TTOASD_Fit_Widths_{q}.png')



for i in range(16):
    for j in range(16):
        tmins = 17

plt.figure(figsize = (9,7))
plt.title(f'Hits v. Threshold DAC\nQinj = {q}')
plt.xlabel('Threshold DAC')
plt.ylabel('Hits')


for pix in fitdata:
    i = int(pix.split('c')[0][1:])
    j = int(pix.split('c')[1])
    
    data = df[(df.row == i)&(df.col==j)]
    x = data.vth
    y = data.hits
    plt.plot(x, y, 'o', label = f'Row {i} Col {j}')
plt.ylim([0, 2])
plt.legend()
plt.savefig(path + f'DAC_v_Hits_multi_{q}.png')


