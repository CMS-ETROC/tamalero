import sqlite3
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use('CMS')

from pathlib import Path

#--------------------------------------------------------------------------#
def convert_dict_to_pandas(input_dict, chip_name, hv, power_mode=None, gain_mode=None):
    bl_nw_df = pd.DataFrame(data = input_dict)
    bl_nw_df['chip_name'] = chip_name
    bl_nw_df['hv'] = hv
    bl_nw_df['power_mode'] = power_mode
    bl_nw_df['gain_mode'] = gain_mode

    return bl_nw_df

#--------------------------------------------------------------------------#
def _build_plot_title(base: str, note: str, timestamp, hv=None) -> str:
    """Build a multi-line plot title: base label, then note (if any), then timestamp/HV."""
    lines = [base]
    if note:
        lines.append(str(note))
    subtitle = str(timestamp)
    if hv is not None:
        subtitle += f" | HV: {hv} V"
    lines.append(subtitle)
    return "\n".join(lines)

#--------------------------------------------------------------------------#
def make_BL_NW_2D_maps(input_df: pd.DataFrame, given_chip_name: str, note: str, save_path: Path, timestamp, hv=None):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use('CMS')

    ## Make BL and NW 2D map
    fig = plt.figure(dpi=200, figsize=(20,10))
    gs = fig.add_gridspec(1,2)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title(_build_plot_title(f"{given_chip_name}: BL (DAC LSB)", note, timestamp, hv), size=17, loc="right")
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
    ax1.set_title(_build_plot_title(f"{given_chip_name}: NW (DAC LSB)", note, timestamp, hv), size=17, loc="right")
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

    for col in range(16):
        for row in range(16):

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


def make_BL_NW_1D_hists(input_df: pd.DataFrame, given_chip_name: str, note: str, save_path: Path, timestamp, hv=None):
    import hist
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    hep.cms.text(loc=0, ax=axes[0], fontsize=17, text="ETL ETROC")
    axes[0].set_title(_build_plot_title(f"{given_chip_name}: BL (DAC LSB)", note, timestamp, hv), size=17, loc="right")
    bl_array = input_df['baseline'].to_numpy().flatten()
    bl_hist = hist.Hist(hist.axis.Regular(128, 0, 1024, name='bl', label='BL [DAC]'))
    bl_hist.fill(bl_array)
    mean, std = bl_array.mean(), bl_array.std()
    bl_hist.plot1d(ax=axes[0], yerr=False, label=f'Mean: {mean:.2f}, Std: {std:.2f}')
    axes[0].legend()

    hep.cms.text(loc=0, ax=axes[1], fontsize=17, text="ETL ETROC")
    axes[1].set_title(_build_plot_title(f"{given_chip_name}: NW (DAC LSB)", note, timestamp, hv), size=17, loc="right")
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

def _ensure_table_columns(sqlconn, table, df):
    """Add any DataFrame columns missing from an existing SQLite table (schema migration)."""
    existing = {row[1] for row in sqlconn.execute(f"PRAGMA table_info({table})")}
    if not existing:
        return  # table doesn't exist yet; to_sql will create it with all columns

    for col, dtype in df.dtypes.items():
        if col in existing:
            continue
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = 'REAL'
        else:
            sql_type = 'TEXT'
        sqlconn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {sql_type}')

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

    timestamp = datetime.datetime.now(datetime.timezone.utc)

    current_df = input_df
    pivot_df = current_df.pivot(index=['row'], columns=['col'], values=['baseline', 'noise_width'])
    hv_value = current_df['hv'].iloc[0] if 'hv' in current_df.columns and not current_df.empty else None

    ### Save baseline into SQL
    current_df.loc[:, "note"] = save_notes
    current_df.loc[:, "saving_timestamp_utc"] = timestamp.replace(tzinfo=None).isoformat(sep=' ', timespec='milliseconds')
    with sqlite3.connect(outfile) as sqlconn:
        _ensure_table_columns(sqlconn, 'baselines', current_df)
        current_df.to_sql('baselines', sqlconn, if_exists='append', index=False)

    print('\n========== Print Baseline and Noise Width ==========')
    print(f'Board name: {chip_name}')
    print(pivot_df.baseline)
    print(pivot_df.noise_width)
    print('  Summary values:')
    print(f"    Baseline mean: {input_df['baseline'].mean():.2f}, std: {input_df['baseline'].std():.2f}")
    print(f"    Noise width mean: {input_df['noise_width'].mean():.2f}, std: {input_df['noise_width'].std():.2f}")

    ## Make BL and NW 2D map
    make_BL_NW_2D_maps(pivot_df, chip_name, save_notes, fig_outdir, timestamp.strftime("%Y-%m-%d_%H-%M"), hv=hv_value)

    ## Make BL and NW 1D hist
    make_BL_NW_1D_hists(current_df, chip_name, save_notes, fig_outdir, timestamp.strftime("%Y-%m-%d_%H-%M"), hv=hv_value)