from tamalero.ETROC import ETROC
from tamalero.ETROC_Emulator import ETROC2_Emulator as software_ETROC2
from tamalero.DataFrame import DataFrame
from tamalero.utils import get_kcu
from tamalero.ReadoutBoard import ReadoutBoard
from tamalero.PixelMask import PixelMask
from tamalero.colors import red, green, yellow
from tamalero.Module import Module
from tamalero.FIFO import FIFO

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import os
import json
import time
from yaml import load, dump
import traceback

import hist
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

# NOTE this should be done
#import logging
#logger = logging.getLogger(__name__)

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def run(ETROC, N, fifo=None):
    # currently uses the software ETROC to produce fake data
    if ETROC.isfake:
        return ETROC2.run(N)
    else:
        fifo.send_l1a(N)
        return fifo.pretty_read(None, raw=True)

def toPixNum(row, col, w=16):
    return col*w+row

def fromPixNum(pix, w=16):
    row = pix%w
    col = int(np.floor(pix/w))
    return row, col

def sigmoid(k,x,x0):
    return 1/(1+np.exp(k*(x-x0)))

# take x,y values and perform fit to sigmoid function
# return steepness(k) and mean(x0)
def sigmoid_fit(x_axis, y_axis):
    res = curve_fit(
        #sigmoid,
        lambda x,a,b: 1/(1+np.exp(a*(x-b))),  # for whatever reason this fit only works with a lambda function?
        x_axis-x_axis[0],
        y_axis,
        maxfev=10000,
        bounds=([-np.inf,-np.inf],[0,np.inf])
    )
    return res[0][0], res[0][1]+x_axis[0]

# parse ETROC dataformat into 1D list of # of hits per pixel
def parse_data(data, N_pix):
    results = np.zeros(N_pix)
    pix_w = int(round(np.sqrt(N_pix)))

    for word in data:
        datatype, res = DF.read(word)
        if datatype == 'data':
            pix = toPixNum(res['row_id'], res['col_id'], pix_w)
            results[pix] += 1

    return results

def vth_scan(ETROC2, vth_min=693, vth_max=709, vth_step=1, decimal=False, fifo=None, absolute=False):
    N_l1a    =  3200 # how many L1As to send
    vth_min  =   vth_min # scan range
    vth_max  =   vth_max
    if not decimal:
        vth_step =   ETROC2.DAC_step # step size
    else:
        vth_step = vth_step
    N_steps  = int((vth_max-vth_min)/vth_step)+1 # number of steps
    N_pix    = 16*16 # total number of pixels

    vth_axis    = np.linspace(vth_min, vth_max, N_steps)
    run_results = np.empty([N_steps, N_pix])
    
    for vth in vth_axis:
        print(f"Working on threshold {vth=}")
        if decimal:
            ETROC2.wr_reg('DAC', int(vth), broadcast=True)
            #print("Acc value", ETROC2.get_ACC(row=0, col=0)) #doesn't work here
        else:
            ETROC2.set_Vth_mV(vth)
        i = int((vth-vth_min)/vth_step)
        run_results[i] = parse_data(run(ETROC2, N_l1a, fifo=fifo), N_pix)
        
    # transpose so each 1d list is for a pixel & normalize
    if absolute:
        run_results = run_results.transpose()
    else:
        run_results = run_results.transpose()/N_l1a
    return [vth_axis.tolist(), run_results.tolist()]

def vth_scan_internal(ETROC2, row=0, col=0, dac_min=0, dac_max=500, dac_step=1):
    N_steps  = int((dac_max-dac_min)/dac_step)+1 # number of steps
    dac_axis = np.linspace(dac_min, dac_max, N_steps)

    ETROC2.setup_accumulator(row, col)
    results = []
    for i in range(dac_min, dac_max+1, dac_step):
        results.append(
            ETROC2.check_accumulator(DAC=i, row=row, col=col)
        )
    run_results = np.array(results)
    return [dac_axis, run_results]

def setup(rb, args):
    modules = rb.modules
    connected_modules = [ mod for mod in modules if mod.connected ]

    if int(args.module) > 0:
        module = modules[int(args.module)-1]
    else:
        module = connected_modules[0]

    print(f"Will proceed with testing the following Module:")
    module.show_status()

    etrocs = []
    masks = []
    for i, etroc in enumerate(module.ETROCs):
        if not args.pixel_masks[i] == 'None':
            mask = PixelMask.from_file(args.pixel_masks[i])
            masked_pixels = mask.get_masked_pixels()

            print(f"\n - Will apply the following pixel mask to ETROC {i}:")
            mask.show()
        masked_pixels = []
        etrocs.append(etroc)
        masks.append(masked_pixels)
    return module, etrocs, masks

def auto_threshold_scan(etroc, args):
    print ("\n - Using auto-threshold calibration for individual pixels")
    print ("Info: if progress is slow, probably most pixel threshold calibrations time out because of high noise levels.")
    baseline = np.empty([16, 16])
    noise_width = np.empty([16, 16])
    with tqdm(total=256, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        for pixel in range(256):
            row = pixel & 0xF
            col = (pixel & 0xF0) >> 4
            baseline[row][col], noise_width[row][col] = etroc.auto_threshold_scan(row=row, col=col, broadcast=False)
            pbar.update()

    print ("Done with threshold scan")
    plot_scan_results(baseline, noise_width, (baseline+noise_width), result_dir, out_dir, mode = 'Auto')
    
    with open(f'{result_dir}/baseline.yaml', 'w') as f:
        dump(baseline.tolist(), f)
     
    with open(f'{result_dir}/noise_width.yaml', 'w') as f:
        dump(noise_width.tolist(), f)
        
    with open(f'{result_dir}/thresholds.yaml', 'w') as f:
        dump((baseline+noise_width).tolist(), f)
        
    return (baseline+noise_width).tolist()
    
def manual_threshold_scan(etroc, fifo, rb_0, args):
    # Prescanning a random pixel to get an idea of the threshold
    row = 4
    col = 3

    prefix = f"module_{etroc.module_id}_etroc_{etroc.chip_no}_"

    elink, slave = etroc.get_elink_for_pixel(row, col)

    rb_0.reset_data_error_count()
    print("\n - Running simple threshold scan on single pixel")
    print(f"Found this pixel on elink {elink}, lpGBT is servant: {slave}")
    vth     = []
    count   = []

    rb_0.get_link_status(elink, slave=slave)

    print("Coarse scan to find the peak location")
    first_val = 1023
    for i in range(0, 1000, 3):
        etroc.wr_reg("DAC", i, row=row, col=col)
        fifo.send_l1a(2000)
        vth.append(i)
        data_cnt = rb_0.read_data_count(elink, slave=slave)
        count.append(data_cnt)
        if data_cnt > 0:
            print(i, data_cnt)
            first_val = i
            if data_cnt == 65535:
                print("Data count is overflowing, breaking coarse scan.")
                break
        rb_0.reset_data_error_count()
        if i > (first_val + 10):
            # break the loop early because this scan is sloooow
            print("I've seen enough, breaking coarse scan.")
            break

    vth_a = np.array(vth)
    count_a = np.array(count)
    vth_max = vth_a[np.argmax(count_a)]
    print(f"Found maximum count at DAC setting vth_max={vth_max}")

    ### threshold scan draft
    dac_min = max(0, vth_max - 75)  # don't run into negatives!
    dac_max = vth_max + 75
    vth_scan_data = vth_scan(
        etroc,
        vth_min = dac_min,
        vth_max = dac_max,
        decimal = True,
        fifo = fifo,
        absolute = True,
    )
    
    vth_axis    = np.array(vth_scan_data[0])
    hit_rate    = np.array(vth_scan_data[1])
    N_pix       = len(hit_rate) # total # of pixels
    N_pix_w     = int(round(np.sqrt(N_pix))) # N_pix in NxN layout
    max_indices = np.argmax(hit_rate, axis=1)
    maximums    = vth_axis[max_indices]
    max_matrix  = np.empty([N_pix_w, N_pix_w])
    noise_matrix  = np.empty([N_pix_w, N_pix_w])

    rawout = {vth_axis[i]:hit_rate.T[i].tolist() for i in range(len(vth_axis))}
    with open(f'{result_dir}/{prefix}thresh_scan_data.json', 'w') as f:
        json.dump(rawout, f)

    threshold_matrix = np.empty([N_pix_w, N_pix_w])
    for pix in range(N_pix):
        r, c = fromPixNum(pix, N_pix_w)
        max_matrix[r][c] = maximums[pix]
        noise_matrix[r][c] = np.size(np.nonzero(hit_rate[pix]))
        max_value = vth_axis[hit_rate[pix]==max(hit_rate[pix])]
        if isinstance(max_value, np.ndarray):
            max_value = max_value[-1]
        zero_dac_values = vth_axis[((vth_axis>(max_value)) & (hit_rate[pix]==0))]
        if len(zero_dac_values)>0:
            threshold_matrix[r][c] = zero_dac_values[0] + 2
        else:
            threshold_matrix[r][c] = dac_max + 2
            
    plot_scan_results(etroc, max_matrix, noise_matrix, threshold_matrix, result_dir, out_dir, mode = 'manual')
    
    #with open(f'{result_dir}/{prefix}thresholds.yaml', 'w') as f:
    #    dump(threshold_matrix.tolist(), f)
    #with open(f'{result_dir}/{prefix}baseline.yaml', 'w') as f:
    #    dump(max_matrix.tolist(), f)
    #with open(f'{result_dir}/{prefix}noise_width.yaml', 'w') as f:
    #    dump(noise_matrix.tolist(), f)
        
    return max_matrix, noise_matrix, vth_scan_data

def plot_scan_results(etroc, max_matrix, noise_matrix, threshold_matrix, result_dir, out_dir, mode = None):
    prefix = f"module_{etroc.module_id}_etroc_{etroc.chip_no}_"

    # 2D histogram of the mean
    # this is based on the code for automatic sigmoid fits
    # for software emulator data below
    N_pix_w = len(max_matrix)
    
    fig, ax = plt.subplots(2,1, figsize=(8,15))
    ax[0].set_title("Peak values of threshold scan")
    ax[1].set_title("Noise width of threshold scan")
    cax1 = ax[0].matshow(max_matrix)
    cax2 = ax[1].matshow(noise_matrix)
    fig.colorbar(cax1,ax=ax[0])
    fig.colorbar(cax2,ax=ax[1])

    ax[0].set_xticks(np.arange(N_pix_w))
    ax[0].set_yticks(np.arange(N_pix_w))

    ax[1].set_xticks(np.arange(N_pix_w))
    ax[1].set_yticks(np.arange(N_pix_w))

    for i in range(N_pix_w):
        for j in range(N_pix_w):
            text = ax[0].text(j, i, int(max_matrix[i,j]),
		        ha="center", va="center", color="w", fontsize="xx-small")

            text1 = ax[1].text(j, i, int(noise_matrix[i,j]),
		        ha="center", va="center", color="w", fontsize="xx-small")

    fig.savefig(f'{result_dir}/{prefix}peak_and_noiseWidth_thresholds.png')
    if args.show_plots:
        plt.show()

    plt.close(fig)
    del fig, ax

    fig, ax = plt.subplots()
    plt.title(f"Thresholds from {mode} scan")
    cax = ax.matshow(threshold_matrix)

    fig.colorbar(cax)
    ax.set_xticks(np.arange(N_pix_w))
    ax.set_yticks(np.arange(N_pix_w))

    for i in range(N_pix_w):
        for j in range(N_pix_w):
            text = ax.text(j, i, int(threshold_matrix[i,j]),
		        ha="center", va="center", color="w", fontsize="xx-small")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.savefig(f'{result_dir}/{prefix}thresholds_manual.png')
    if args.show_plots:
        plt.show()

    plt.close(fig)
    del fig, ax

def isolate(etrocs, n):
    for i, e in enumerate(etrocs):
        if e.is_connected():
            e.wr_reg("disDataReadout", 0x0, broadcast=True)
            if i != n:
                e.wr_reg("disDataReadout", 0x1, broadcast=True)

def check_temp(etroc):
    temp = etroc.check_temp()
    return temp

def readout_tests(etroc, masked_pixels, rb_0, args, result_dir = None, out_dir = None):
    etroc.deactivate_hot_pixels(pixels=masked_pixels)

    df = DataFrame()
    fifo = FIFO(rb=rb_0)
    
    print("\n - Checking elinks")
    start_time = time.time()
    temperatures = []
    while True:
        try:
            fifo.reset()
            fifo.send_l1a(10)
            _ = fifo.pretty_read(df)
            fifo.reset()
            break
        except:
            print("Initial (re)set of FIFO.")
            if time.time() - start_time > 1:
                print("FIFO state is unexpected.")
                raise
    etroc.reset()
    fifo.reset()

    tmp_temp = check_temp(etroc)
    temperatures.append((time.time(), tmp_temp))
    print('Current ETROC Temperature Reading:', tmp_temp)
    
    etroc.wr_reg("DAC", 0, broadcast=True)  # make sure that we're not additionally picking up any noise
    etroc.wr_reg("selfTestOccupancy", 2, broadcast=True)
    etroc.wr_reg("workMode", 0x1, broadcast=True)
    # maybe need to reset??
    etroc.deactivate_hot_pixels(pixels=masked_pixels)
    etroc.wr_reg("onChipL1AConf", 0x2)  # NOTE: internal L1A is around 1MHz, so we're only turning this on for the shortest amount of time.
    time.sleep(0.1)
    etroc.wr_reg("onChipL1AConf", 0x0)
    print("FIFO occupancy:", fifo.get_occupancy())
    test_data = []
    while fifo.get_occupancy() > 0:
        test_data += fifo.pretty_read(df)

    hits_total = np.zeros((16,16))
    row_axis = hist.axis.Regular(16, -0.5, 15.5, name="row", label="row")
    col_axis = hist.axis.Regular(16, -0.5, 15.5, name="col", label="col")
    hit_matrix = hist.Hist(col_axis,row_axis)
    n_events_total = 0
    n_events_hit   = 0
    n_events_err   = 0
    for d in test_data:
        #print(d[1]['raw'])
        if d[0] == 'trailer':
            n_events_total += 1
            if d[1]['hits'] > 0:
                n_events_hit += 1
        if d[0] == 'data':
            hit_matrix.fill(row=d[1]['row_id'], col=d[1]['col_id'])
            hits_total[d[1]['row_id']][d[1]['col_id']] += 1
            if d[1]['row_id'] != d[1]['row_id2']:
                print("Unpacking error in row ID")
                n_events_err += 1
            if d[1]['col_id'] != d[1]['col_id2']:
                print("Unpacking error in col ID")
                n_events_err += 1
            if d[1]['test_pattern'] != 0xaa:
                print(f"Unpacking error in test pattern, expected 0xAA but got {d[1]['test_pattern']=}")
                n_events_err += 1

    print(f"Got number of total events {n_events_total=}")
    print(f"Events with at least one hit {n_events_hit=}")
    print(f"Events with some error in data unpacking {n_events_err=}")

    fig, ax = plt.subplots(1,1,figsize=(7,7))
    hit_matrix.plot2d(
        ax=ax,
    )
    ax.set_ylabel(r'$Row$')
    ax.set_xlabel(r'$Column$')
    hep.cms.label(
            "ETL Preliminary",
            data=True,
            lumi='0',
            com=0,
            loc=0,
            ax=ax,
            fontsize=15,
        )
    prefix = f'module_{etroc.module_id}_etroc_{etroc.chip_no}_'
    name = prefix+'hit_matrix_internal_test_pattern'
    fig.savefig(os.path.join(result_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(result_dir, f"{name}.png"))
    if args.show_plots:
        plt.show()
    plt.close()

    fifo.reset()
    print("\n - Testing fast command communication - Sending two L1As")
    fifo.send_l1a(2)
    for x in fifo.pretty_read(df):
        print(x)

    #etroc.QInj_unset(broadcast=True)
    fifo.reset()

    print("Will use workMode 1 to get some occupancy (no noise or charge injection)")
    etroc.wr_reg("workMode", 0x1, broadcast=True)  # this overwrites disDataReadout!
    etroc.deactivate_hot_pixels(pixels=masked_pixels)
    #etroc.deactivate_hot_pixels(pixels=[(i,1) for i in range(16)])

    for j in range(1):
        # One go is enough, but can run through this loop many times if there's any issue
        # with FIFO / data readout at any point
        print(j)
        ### Another occupancy map
        i = 0
        occupancy = 0
        print("\n - Will send L1As until FIFO is full.")

        #etroc.QInj_set(30, 0, row=3, col=3, broadcast=False)
        start_time = time.time()
        with tqdm(total=65536, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            while not fifo.is_full():
                fifo.send_l1a()
                #fifo.send_QInj(delay=j)
                #fifo.send_QInj()
                i +=1
                if i%100 == 0:
                    tmp = fifo.get_occupancy()
                    pbar.update(tmp-occupancy)
                    occupancy = tmp
                #if time.time()-start_time>5:
                #    print("Time out")
                #    break

        test_data = []
        while fifo.get_occupancy() > 0:
            test_data += fifo.pretty_read(df)

        hits_total = np.zeros((16,16))
        hit_matrix = hist.Hist(col_axis,row_axis)
        n_events_total = 0
        n_events_hit   = 0
        for d in test_data:
            if d[0] == 'trailer':
                n_events_total += 1
                if d[1]['hits'] > 0:
                    n_events_hit += 1
            if d[0] == 'data':
                hit_matrix.fill(row=d[1]['row_id'], col=d[1]['col_id'])
                hits_total[d[1]['row_id']][d[1]['col_id']] += 1
                # NOTE could do some CRC check.

        print(f"Got number of total events {n_events_total=}")
        print(f"Events with at least one hit {n_events_hit=}")

        fig, ax = plt.subplots(1,1,figsize=(7,7))
        hit_matrix.plot2d(
            ax=ax,
        )
        ax.set_ylabel(r'$Row$')
        ax.set_xlabel(r'$Column$')
        hep.cms.label(
                "ETL Preliminary",
                data=True,
                lumi='0',
                com=0,
                loc=0,
                ax=ax,
                fontsize=15,
            )
        name = prefix+'hit_matrix_external_L1A'
        fig.savefig(os.path.join(result_dir, "{}.pdf".format(name)))
        fig.savefig(os.path.join(result_dir, "{}.png".format(name)))
        if args.show_plots:
            plt.show()
        plt.close()

        print("\nOccupancy vs column:")
        hit_matrix[{"row":sum}].show(columns=100)
        print("\nOccupancy vs row:")
        hit_matrix[{"col":sum}].show(columns=100)

    # Set the chip back into well-defined workMode 0
    etroc.wr_reg("workMode", 0x0, broadcast=True)

    tmp_temp = check_temp(etroc)
    temperatures.append((time.time(), tmp_temp))
    print('Current ETROC Temperature Reading:', tmp_temp)
    if args.threshold == 'auto':
        baseline, noise_width = etroc.run_threshold_scan()
        etroc.plot_threshold(outdir=result_dir, noise_width=False)
        etroc.plot_threshold(outdir=result_dir, noise_width=True)
    elif args.threshold == "manual":
        baseline, noise_width, vth_scan_result = manual_threshold_scan(etroc, fifo, rb_0, args)
    else:
        raise NotImplementedError("This option is currently not expected to work")
        print(f"Trying to load tresholds from the following file: {args.threshold}")
        with open(args.threshold, 'r') as f:
            threshold_matrix = load(f)

        for row in range(16):
            for col in range(16):
                etroc.wr_reg('DAC', int(threshold_matrix[row][col]), row=row, col=col)
    tmp_temp = check_temp(etroc)
    temperatures.append((time.time(), tmp_temp))
    print('Current ETROC Temperature Reading:', tmp_temp)

    threshold_matrix = baseline+noise_width
    # store results
    with open(f'{result_dir}/{prefix}baseline_{args.threshold}.yaml', 'w') as f:
        dump(baseline.tolist(), f)

    with open(f'{result_dir}/{prefix}noise_width_{args.threshold}.yaml', 'w') as f:
        dump(noise_width.tolist(), f)

    with open(f'{result_dir}/{prefix}thresholds_{args.threshold}.yaml', 'w') as f:
        dump((baseline+noise_width).tolist(), f)

    with open(f'{result_dir}/temperatures.log', 'w') as f:
        for t_log, temp_log in temperatures:
            f.writelines(f'{t_log}, {temp_log}\n')

    if args.threshold ==  'manual':
        return threshold_matrix, vth_scan_result
    else:
        return threshold_matrix

def qinj(etroc, mask, rb, thresholds, out_dir, result_dir, args):
    df = DataFrame()
    fifo = FIFO(rb=rb)

    fifo.reset()
    delay = 10
    i = int(args.row)
    j = int(args.col)
    L1Adelay = 501
    RBL1Adelay = 504
    
    if args.threshold == 'auto':
        etroc.bypass_THCal()
    print('Starting Charge Injection run using the following configurations')
    if len(args.vth_axis) < 2:
        vth_axis = range(int(np.max([np.min(thresholds) - 20, 0])), int(np.min([np.max(thresholds) + 350, 1000]))) 
    elif len(args.vth_axis) == 2:
        vth_axis = range(*args.vth_axis)
    else:
        vth_axis = np.linspace(*args.vth_axis)
    print(f'DAC Range {np.min(vth_axis)} {np.max(vth_axis)}')
    
    if len(args.charges) == 0:
       charges = [5, 15, 30]
    else:
       charges = args.charges
    print(f'Charges {charges}')
    
    print(f'No. of L1As: {args.nl1a}')
    print(f'Pixel Row {args.row} Col {args.col}')
    print(f'Charge Injection Delay {delay}')
    print(f'L1A Injection Delay {L1Adelay}')
    
    results =[[] for i in range(0,len(charges))]
    TOA = [[] for i in range(0,len(charges))]
    TOT =  [[] for i in range(0,len(charges))]
    CAL = [[] for i in range(0,len(charges))]
    k=0
    
    for q in charges:
        print(f"\n - Will send L1a/QInj pulse with delay of {delay} cycles and charge of {q} fC")
        print(f"\n - to pixel at Row {i}, Col {j}.")
        for vth in tqdm(vth_axis):
            worked = False
            qinjatt = 0
            while(not worked) and (qinjatt < 10):
                try:
                    etroc.QInj_set(q, delay, L1Adelay, row=i, col=j, broadcast = False) #set reg on ETROC
                    etroc.wr_reg('DAC', int(vth), row=i, col=j, broadcast=False) #set vth on ETROC
                    #print(etroc.rd_reg('DAC', row=i, col=j)) #read vth on ETROC
                    fifo.send_QInj(count=int(args.nl1a), delay=RBL1Adelay) #send Qinj pulses with L1Adelay
                    result = fifo.pretty_read(df)
                    worked = True
                except KeyboardInterrupt:
                    print(f'Interupted by keyboard. Terminating run')
                    raise
                except:
                    print(f'Attempt Number {qinjatt} failed. Trying again.')
                    print(traceback.format_exc())
                qinjatt += 1
            hits=0
            toa=[]
            tot=[]
            cal=[]
            for word in result:
                if(word[0] == 'data'):
                    toa.append(int(word[1]['toa']))
                    tot.append(int(word[1]['tot']))
                    cal.append(int(word[1]['cal']))
                if(word[0] == 'trailer'):
                    hits+=word[1]['hits']
            results[k].append(int(hits))
            TOT[k].append(tot)
            TOA[k].append(toa)
            CAL[k].append(cal)
        k+=1
        scan_df = {'vth': list(vth_axis),
            'hits': list(results[k-1]),
            'toa' : TOA[k-1],
            'tot' : TOT[k-1],
            'cal' : CAL[k-1]}

        with open(f"{out_dir}/Qinj_scan_ETROC_{etroc.chip_no}_L1A_{L1Adelay}_{q}.json", 'w') as f:
            json.dump(scan_df, f)
        
    fig, ax = plt.subplots()
    
    plt.title("S curve for Qinj")
    plt.xlabel("Vth")
    plt.ylabel("hit rate")
    for i in range(0,len(charges)):
        plt.plot(vth_axis, results[i], '.-')
                
    #plt.xlim(410,820)
    plt.grid(True)
    plt.legend(loc='best')
              
    fig.savefig(f'{result_dir}/Scurve_Qinj_ETROC_{etroc.chip_no}.png')
    if args.show_plots:
        plt.show()
    plt.close(fig)
    del fig, ax

# initiate
ETROC2 = software_ETROC2()  # currently using Software ETROC2 (fake)
print("ETROC2 emulator instantiated, base configuration successful")
DF = DataFrame('ETROC2')

if __name__ == '__main__':
    import argparse
    argParser = argparse.ArgumentParser(description = "Argument parser")
    #Setup Options
    argParser.add_argument('--configuration', action='store', default='modulev1', choices=['modulev0b', 'modulev1'], help="Board configuration to be loaded")
    argParser.add_argument('--kcu', action='store', default='192.168.0.10', help="IP Address of KCU105 board")
    argParser.add_argument('--host', action='store', default='localhost', help="Hostname for control hub")
    argParser.add_argument('--module', action='store', default=1, choices=['1','2','3'], help="Module to test")
    argParser.add_argument('--enable_power_board', action='store_true', help="Enable Power Board (all modules). Jumpers must still be set as well.")
    argParser.add_argument('--moduleid', action='store', default=0, help="")
    argParser.add_argument('--multi', action='store_true', help="Run multiple modules at once (for data taking only!)")
    argParser.add_argument('--etrocs', action = 'store', nargs = '*', type = int, default = [0], help = 'ETROC to use on a multi-ETROC module')
    argParser.add_argument('--all-etrocs', action = 'store', type = int, default = 0, help = 'ETROC to use on a multi-ETROC module')
    argParser.add_argument('--run_tag', action = 'store', default = None, help = 'descriptive tag to identify run beyond timestamp')
    argParser.add_argument('--rb', action='store', default=0, type = int, help="")
    argParser.add_argument('--reset_fifo', action='store_true')
    #Task Set
    argParser.add_argument('--test_chip', action='store_true', default=False, help="Test simple read/write functionality for real chip?")
    argParser.add_argument('--config_chip', action='store_true')    
    #run options
    argParser.add_argument('--hard_reset', action='store_true', help="")
    argParser.add_argument('--test_data_stream', action='store_true', help="")
    argParser.add_argument('--mode', action='store', default='dual', choices=['dual', 'single'], help="Port mode for ETROC2")
    argParser.add_argument('--skip_sanity_checks', action='store_true', default=False, help="Don't run sanity checks of ETROC2 chip")
    argParser.add_argument('--threshold', action='store', default='auto', help="Use thresholds from manual or automatic scan?")
    argParser.add_argument('--row', action='store', default=4, help="Pixel row to be tested")
    argParser.add_argument('--col', action='store', default=3, help="Pixel column to be tested")
    argParser.add_argument('--pixel_masks', action='store', nargs = '*', default=['None'], help="Pixel mask to apply")
    argParser.add_argument('--show_plots', action = 'store_true')
    argParser.add_argument('--external_vref', action = 'store_true')
    argParser.add_argument('--run_internal', action = 'store_true')
    #Charge injection
    argParser.add_argument('--qinj', action='store_true')
    argParser.add_argument('--charges', action = 'store', type = int, nargs = '*', default = [])
    argParser.add_argument('--vth_axis', action = 'store', type = int, nargs = '*', default = [])
    argParser.add_argument('--nl1a', action = 'store', type = int, default = 3200)
    #argParser.add_argument('--pixels', action = 'store', default = [])
    #charges
    #vth_axis
    #pixels
    #nl1a
    argParser.add_argument('--config', action='store', default=None, help="config file to use")
    args = argParser.parse_args()

    if len(args.pixel_masks)==1 and args.pixel_masks[0] == 'None':
        args.pixel_masks = ['None']*12

    assert int(args.moduleid)>0, "Module ID is not specified. This is a new feature, please run with --moduleid MODULEID, where MODULEID is the number on the module test board."
    MID = int(args.moduleid)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    if args.run_tag:
        timestamp += '_' + args.run_tag
    result_dir = f"results/{MID}/{timestamp}/" #MultiETROC
    out_dir = f"outputs/{MID}/{timestamp}/"
 
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print(f"Will test module with ID {MID}.") #MultiETROC
    print(f"All results will be stored under timestamp {timestamp}")
    
    if len(args.pixel_masks) == 1 and len(args.etrocs) > 1:
        args.pixel_masks *= len(args.etrocs)

    start_time = time.time()
    print("Getting KCU")
    kcu = get_kcu(args.kcu, control_hub=True, host=args.host, verbose=False)
    int_time = time.time()
    print("Getting RB")
    rb = ReadoutBoard(args.rb, kcu=kcu, config=args.configuration)
    int2_time = time.time()
    print("Connecting modules")
    moduleids = [0,0,0]
    moduleids[int(args.module)-1] = MID
    rb.connect_modules(moduleids=moduleids, hard_reset=True, ext_vref=args.external_vref)
    for mod in rb.modules:
        mod.show_status()

    if args.reset_fifo:
        print("RESETTING FIFO")
        fifo = FIFO(rb)
        fifo.get_occupancy()
        fifo.set_trigger_rate(0)
        fifo.reset()

    if args.test_data_stream:
        print("Testing data stream now...")
        for mod in rb.modules:
            if mod.connected:
                for etroc in mod.ETROCs:
                    if etroc.is_connected():
                        print(f"Found connected ETROC {etroc.chip_no} on module {etroc.module_id}")
                        etroc.test_config(occupancy=0)
        fifo = FIFO(rb)
        df = DataFrame('ETROC2')

        fifo.send_l1a(1)
        fifo.reset()

        fifo.send_l1a(1)
        for x in fifo.pretty_read(df):
            print(x)
        print("Done")

    end_time = time.time()

    print("KCU init done in     {:.2f}s".format(int_time-start_time))
    print("RB init done in      {:.2f}s".format(int2_time-int_time))
    print("Module init done in  {:.2f}s".format(end_time-int2_time))  # default config is what's slow

    print("ADC status:")
    if rb.ver < 3:
        rb.SCA.read_adcs()
    else:
        rb.MUX64.read_channels()
    rb.DAQ_LPGBT.read_adcs()

    if args.test_chip:
        module, etrocs, masks = setup(rb, args)
        #Check all etrocs off?
        for i, etroc in enumerate(etrocs):
            if etroc.is_connected():
                print(f"Testing ETROC {etroc.chip_no} on Module {etroc.module_id} now")
                mask = masks[i]
                if not args.skip_sanity_checks:
                    etroc.pixel_sanity_check()
                isolate(etrocs, i)
                thresholds = readout_tests(etroc, mask, rb, args, result_dir =  result_dir, out_dir = out_dir)
                if args.threshold == 'manual' and args.run_internal:
                    print("RUNNING THRESHOLD SCAN")
                    os.makedirs(f'{result_dir}/pixels/')
                    baseline, noise_width = etroc.run_threshold_scan()
                    thresholds, vth_scan_results = thresholds
                    dac_ext, res_ext = vth_scan_results
                    with tqdm(total=256, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                        for pixel in range(256):
                            row = pixel & 0xF
                            col = (pixel & 0xF0) >> 4

                            #res_ext_pixel =
                            dac_mode_index = res_ext[pixel].index(max(res_ext[pixel]))
                            dac_mode = int(dac_ext[dac_mode_index])

                            dac_min = dac_mode - 30
                            dac_max = dac_mode + 30

                            dac, res = vth_scan_internal(
                                etroc,
                                row=row,
                                col=col,
                                dac_min=dac_min,
                                dac_max=dac_max,
                                dac_step=1,
                            )
                            res_normalized = res/max(res)

                            res_ext_normalized = np.array(res_ext[pixel])/max(res_ext[pixel])

                            fig, ax = plt.subplots()
                            plt.title(f"S-curve for pixel ({row},{col})")
                            ax.plot(dac, res_normalized, '.-', color='blue', label='internal (acc)')
                            ax.plot(dac, res_ext_normalized[dac_mode_index-30:dac_mode_index+31], '.-', color='red', label='external')

                            ax.set_ylim(0, 1.05)
                            ax.set_xlim(dac_min, dac_max)
                            ax.set_xlabel("DAC")
                            ax.set_ylabel("normalized count")
                            BL = int(baseline[row][col])
                            NW = int(noise_width[row][col])
                            ax.vlines(x=BL, ymin=0, ymax=1.05, color='green', label=f'BL +/- NW: {BL} +/- {NW}', lw=2)
                            ax.vlines(x=BL+NW, ymin=0, ymax=1.05, color='green', lw=1, ls=':')
                            ax.vlines(x=BL-NW, ymin=0, ymax=1.05, color='green', lw=1, ls=':')
                            plt.legend()
                            fig.savefig(f'{result_dir}/pixels/scan_internal_row_{row}_col_{col}.png')

                            plt.close()
                            del fig, ax

                            pbar.update()

                if args.qinj:
                    qinj(etroc, mask, rb, thresholds, out_dir, result_dir, args)
            else:
                    print('ETROC', i, 'is not connected')
