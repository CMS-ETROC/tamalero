#!/usr/bin/env python3

import os
import json
import awkward as ak
import numpy as np
import argparse
import hist

if __name__ == '__main__':

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument("--first_run", action='store', type=int, help="Number of the run to start with")
    argParser.add_argument("--last_run", action='store', type=int, help="Number of the run to end with")
    argParser.add_argument("--rb", action='store', type=int, default=0, help="RB to run")
    argParser.add_argument("--specific_runs", action='store', nargs="+", help="List of specific runs to stack, formated xxx yyy zzz. Can handle an arbitrary number of runs, user is recommened not to use more than a few")
    argParser.add_argument("--module", action='store', default='37', help="Module number for the runs")
    args = argParser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))

files_to_stack = []
stacked = ak.Array([])

fail_string="_missing"
fail_list=[]

#TODO: reconfigure to work with glob for better flexibility in names
if args.specific_runs:
    for run in args.specific_runs:
        with open("{}/../ETROC_output/output_run_{}_rb{}.json".format(here, run, args.rb), "r") as f:
            files_to_stack.append(ak.from_json(json.load(f)))
        print("Run {} contains {} events".format(run,len(files_to_stack[-1])))
else:
    for run in range(args.first_run, args.last_run+1):
        try:
            with open("{}/../ETROC_output/output_run_{}_rb{}.json".format(here, run, args.rb), "r") as f:
                files_to_stack.append(ak.from_json(json.load(f)))
            print("Run {} contains {} events".format(run,len(files_to_stack[-1])))
        except:
            print("Run {} failed to be added to the stack, will continue without it".format(run))
            fail_string+="_{}".format(run)
            fail_list.append(run)

stacked = ak.concatenate(files_to_stack)
print("The total stack contains {} events".format(len(stacked)))

if args.specific_runs:
    specific_name=f"{here}/../ETROC_output/output_run_"
    for run in args.specific_runs:
        specific_name+=(str(run)+"_")
    specific_name+="stacked"
    with open(specific_name+".json", "w") as f:
            json.dump(ak.to_json(stacked), f)
else:
    stacked_name="{}/../ETROC_output/module_{}_output_run_{}_to_{}_stacked".format(here, args.module, args.first_run,args.last_run)+fail_string
    with open(stacked_name+".json", "w") as f:
            json.dump(ak.to_json(stacked), f)

# make some plots
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

hits = np.zeros([16, 16])

if args.module == '36':
    masked = [(3,8),(15,15)]
elif args.module == '38':
    masked = [(11,5)]
elif args.module == '37':
    masked = [(4,0)]
elif args.module == '39':
    masked = [(11,15),(12,8),(12,15),(13,15),(14,15),(15,12),(15,14)]
elif args.module == '106':
    #masked = [(12,2),(4,15),(5,15)]
    masked = [(10,4),(5,15)]
    #masked = []
elif args.module == '111':
    masked = [(3,6)]
else:
    masked = []


row_axis = hist.axis.Integer(0, 15, name='row', label=r"row")
col_axis = hist.axis.Integer(0, 15, name='col', label=r"col")

hit_hist = hist.Hist(row_axis, col_axis)

for event in stacked:
    for row, col in zip(event.row, event.col):
        if not (row, col) in masked:
            hits[row][col] += 1
            hit_hist.fill(row=row, col=col)

total_hits = np.sum(hits)
print(f"Total number of hits: {total_hits}")

fig, ax = plt.subplots(1,1,figsize=(15,15))
cax = ax.matshow(hits)
ax.set_title(f'Module {args.module}')
ax.set_ylabel(r'$Row$')
ax.set_xlabel(r'$Column$')
ax.margins(0.05)
for i in range(16):
    for j in range(16):
        text = ax.text(j, i, int(hits[i,j]),
                ha="center", va="center", color="w", fontsize="xx-small")

out_name = specific_name if args.specific_runs else stacked_name

fig.colorbar(cax,ax=ax, label='Hits')
fig.savefig(out_name+".pdf")
fig.savefig(out_name+".png")


fig, ax = plt.subplots(1,1,figsize=(15,15))
hit_hist[{'row':sum}].plot1d(ax=ax)
fig.savefig(out_name+"_col.png")


fig, ax = plt.subplots(1,1,figsize=(15,15))
hit_hist[{'col':sum}].plot1d(ax=ax)
fig.savefig(out_name+"_row.png")
