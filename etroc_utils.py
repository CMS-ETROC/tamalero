import sqlite3
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use('CMS')

from pathlib import Path

#--------------------------------------------------------------------------#
def convert_dict_to_pandas(input_dict, chip_name):
    bl_nw_df = pd.DataFrame(data = input_dict)
    bl_nw_df['chip_name'] = chip_name

    return bl_nw_df

#--------------------------------------------------------------------------#
def make_BL_NW_2D_maps(input_df: pd.DataFrame, given_chip_name: str, note: str, save_path: Path, timestamp):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use('CMS')

    ## Make BL and NW 2D map
    fig = plt.figure(dpi=200, figsize=(20,10))
    gs = fig.add_gridspec(1,2)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title(f"{given_chip_name}: BL (DAC LSB)\n{note}", size=17, loc="right")
    img0 = ax0.imshow(input_df.baseline, interpolation='none', vmin=input_df.baseline.to_numpy().reshape(-1).min(), vmax=input_df.baseline.to_numpy().reshape(-1).max())
    ax0.set_aspect("equal")
    ax0.invert_xaxis()
    ax0.invert_yaxis()
    plt.xticks(range(16), range(16), rotation="vertical")
    plt.yticks(range(16), range(16))
    hep.cms.text(loc=0, ax=ax0, fontsize=17, text="ETL ETROC")
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax, orientation="vertical")

    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title(f"{given_chip_name}: NW (DAC LSB)\n{note}", size=17, loc="right")
    img1 = ax1.imshow(input_df.noise_width, interpolation='none', vmin=0, vmax=16)
    ax1.set_aspect("equal")
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    plt.xticks(range(16), range(16), rotation="vertical")
    plt.yticks(range(16), range(16))
    hep.cms.text(loc=0, ax=ax1, fontsize=17, text="ETL ETROC")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax, orientation="vertical")

    bl_threshold = 0.55 * (input_df.baseline.values.max() - input_df.baseline.values.min()) + input_df.baseline.values.min()

    # for col in range(16):
    #     for row in range(16):
    for col in range(2):
        for row in range(2):

            bl_value = int(input_df.baseline[col][row])
            nw_value = int(input_df.noise_width[col][row])
            bl_text_color = 'black' if bl_value > bl_threshold else 'white'
            nw_text_color = 'black' if nw_value > 9 else 'white'

            ax0.text(col,row, bl_value, c=bl_text_color, size=10, rotation=45, fontweight="bold", ha="center", va="center")
            ax1.text(col,row, nw_value, c=nw_text_color, size=11, rotation=45, fontweight="bold", ha="center", va="center")

    plt.tight_layout()
    board_dir = save_path / given_chip_name
    board_dir.mkdir(exist_ok=True)
    fig.savefig(board_dir / f'{given_chip_name}_BL_NW_2D_map_{timestamp}.png')


def make_BL_NW_1D_hists(input_df: pd.DataFrame, given_chip_name: str, note: str, save_path: Path, timestamp):
    import hist
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    hep.cms.text(loc=0, ax=axes[0], fontsize=17, text="ETL ETROC")
    axes[0].set_title(f"{given_chip_name}: BL (DAC LSB)\n{note}", size=17, loc="right")
    bl_array = input_df['baseline'].to_numpy().flatten()
    bl_hist = hist.Hist(hist.axis.Regular(128, 0, 1024, name='bl', label='BL [DAC]'))
    bl_hist.fill(bl_array)
    mean, std = bl_array.mean(), bl_array.std()
    bl_hist.plot1d(ax=axes[0], yerr=False, label=f'Mean: {mean:.2f}, Std: {std:.2f}')
    axes[0].legend()

    hep.cms.text(loc=0, ax=axes[1], fontsize=17, text="ETL ETROC")
    axes[1].set_title(f"{given_chip_name}: NW (DAC LSB)\n{note}", size=17, loc="right")
    nw_hist = hist.Hist(hist.axis.Regular(16, 0, 16, name='nw', label='NW [DAC]'))
    nw_array = input_df['noise_width'].to_numpy().flatten()
    nw_hist.fill(nw_array)
    mean, std = nw_array.mean(), nw_array.std()
    nw_hist.plot1d(ax=axes[1], yerr=False, label=f'Mean: {mean:.2f}, Std: {std:.2f}')
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1].xaxis.set_minor_locator(ticker.NullLocator())
    axes[1].legend()

    plt.tight_layout()
    board_dir = save_path / given_chip_name
    board_dir.mkdir(exist_ok=True)
    fig.savefig(board_dir / f'{given_chip_name}_BL_NW_1D_hist_{timestamp}.png')

#--------------------------------------------------------------------------#
def save_baselines(
        input_df: pd.DataFrame,
        chip_name: str,
        hist_dir: str = "../ETROC-History",
        fig_dir: str = '../ETROC-figures',
        save_notes: str = "",
    ):

    save_mother_path = Path(hist_dir)
    save_mother_path.mkdir(exist_ok=True, parents=True)
    outfile = save_mother_path / 'BaselineHistory.sqlite'

    fig_outdir = Path(fig_dir)
    fig_outdir = fig_outdir / (datetime.date.today().isoformat() + '_Testing_Plots')
    fig_outdir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    current_df = input_df
    pivot_df = current_df.pivot(index=['row'], columns=['col'], values=['baseline', 'noise_width'])

    ### Save baseline into SQL
    current_df.loc[:, "save_notes"] = save_notes
    with sqlite3.connect(outfile) as sqlconn:
        current_df.to_sql('baselines', sqlconn, if_exists='append', index=False)

    ## Make BL and NW 2D map
    make_BL_NW_2D_maps(pivot_df, chip_name, save_notes, fig_outdir, timestamp)

    ## Make BL and NW 1D hist
    make_BL_NW_1D_hists(current_df, chip_name, save_notes, fig_outdir, timestamp)