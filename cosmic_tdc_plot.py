import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tamalero.FIFO import merge_words
from tamalero.DataFrame import DataFrame


def read_dat_files(data_dir):
    data_path = Path(data_dir)
    dat_files = sorted(data_path.glob('file_*_CE.dat'))
    
    if not dat_files:
        print(f"No .dat files found")
        return []
    
    print(f"Found {len(dat_files)} .dat files")
    all_words = []
    for dat_file in dat_files:
        with open(dat_file, 'rb') as f:
            data = f.read()
        num_words = len(data) // 4
        words = struct.unpack(f'<{num_words}I', data)
        all_words.extend(words)
    
    print(f"Total words: {len(all_words)}")
    return all_words

def parse_etroc_data(raw_words):
    df = DataFrame('ETROC2')
    merged_64bit = merge_words(raw_words)
    
    hits = []
    for word in merged_64bit:
        data_type, parsed_data = df.read(word, quiet=True)
        
        if data_type == 'data':
            elink = parsed_data.get('elink')
            row = parsed_data.get('row_id')
            col = parsed_data.get('col_id')
            toa = parsed_data.get('toa')
            tot = parsed_data.get('tot')
            cal = parsed_data.get('cal')
            
            if all(v is not None for v in [elink, row, col, toa, tot, cal]):
                if 0 <= row < 16 and 0 <= col < 16:
                    hits.append({
                        'elink': int(elink),
                        'row': int(row),
                        'col': int(col),
                        'toa': int(toa),
                        'tot': int(tot),
                        'cal': int(cal)
                    })
    
    print(f"Valid hits: {len(hits)}")
    return hits

def plot_tdc_analysis(hits, output_dir, chip_name="ETROC"):
    # Organize by chip
    elink_to_chip = {0: 0, 4: 1, 8: 2, 12: 3}
    chip_names = {
        0: "ET2.02-PT-IH6",
        1: "ET2.02-PT-IH9",
        2: "ET2.02-PT-IH68",
        3: "Chip-4"
    }
    active_elinks = set(h['elink'] for h in hits)
    
    for elink in sorted(active_elinks):
        if elink not in elink_to_chip:
            continue
        
        chip_id = elink_to_chip[elink]
        chip_hits = [h for h in hits if h['elink'] == elink]

        toa_data = np.array([h['toa'] for h in chip_hits])
        tot_data = np.array([h['tot'] for h in chip_hits])
        cal_data = np.array([h['cal'] for h in chip_hits])
        row_data = np.array([h['row'] for h in chip_hits])
        col_data = np.array([h['col'] for h in chip_hits])

        fig = plt.figure(figsize=(20, 12))
        # plt.subplots_adjust(top=0.95)
        fig.suptitle(f'{chip_name[chip_id]} (E-link {elink}) - TDC Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Heat Map
        ax1 = plt.subplot(2, 3, 1)
        heatmap = np.zeros((16, 16))
        for h in chip_hits:
            heatmap[h['row'], h['col']] += 1
        im1 = ax1.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_title('Heat Map', fontsize=14)
        ax1.set_xlabel('COL')
        ax1.set_ylabel('ROW')
        ax1.invert_xaxis()

        for row in range(16):
            for col in range(16):
                count = int(heatmap[row, col])
                if count > 0:  
                    text_color = 'white' if count < heatmap.max() * 0.6 else 'black'
                    ax1.text(col, row, f'{count}', 
                           ha='center', va='center', 
                           color=text_color, fontsize=7)
        
        plt.colorbar(im1, ax=ax1, label='Hits')
        
        # 2. CAL Distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(cal_data, bins=50, range=(150, 250), histtype='step', lw=2, color='blue')
        ax2.set_xlabel('CAL [LSB]')
        ax2.set_ylabel('Counts')
        ax2.set_title('CAL Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, f'Mean: {cal_data.mean():.1f}\nStd: {cal_data.std():.1f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. TOT Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(tot_data, bins=64, range=(0, 512), histtype='step', lw=2, color='red')
        ax3.set_xlabel('TOT [LSB]')
        ax3.set_ylabel('Counts')
        ax3.set_title('TOT Distribution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.text(0.05, 0.95, f'Mean: {tot_data.mean():.1f}\nStd: {tot_data.std():.1f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. TOA Distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(toa_data, bins=64, range=(0, 1024), histtype='step', lw=2, color='green')
        ax4.set_xlabel('TOA [LSB]')
        ax4.set_ylabel('Counts')
        ax4.set_title('TOA Distribution', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.text(0.05, 0.95, f'Mean: {toa_data.mean():.1f}\nStd: {toa_data.std():.1f}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. TOA vs TOT 2D histogram
        ax5 = plt.subplot(2, 3, 5)
        h2d, xedges, yedges = np.histogram2d(toa_data, tot_data, 
                                             bins=[64, 64], 
                                             range=[[0, 1024], [0, 512]])
        im5 = ax5.imshow(h2d.T, origin='lower', aspect='auto', cmap='hot',
                        extent=[0, 1024, 0, 512])
        ax5.set_xlabel('TOA [LSB]')
        ax5.set_ylabel('TOT [LSB]')
        ax5.set_title('TOA vs TOT', fontsize=14)
        plt.colorbar(im5, ax=ax5, label='Counts')
        
        # 6. TOA vs TOT with marginal distributions
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(toa_data, tot_data, alpha=0.3, s=1, c=cal_data, cmap='viridis')
        ax6.set_xlabel('TOA [LSB]')
        ax6.set_ylabel('TOT [LSB]')
        ax6.set_title('TOA vs TOT Scatter (color=CAL)', fontsize=14)
        ax6.set_xlim(0, 1024)
        ax6.set_ylim(0, 512)
        cbar = plt.colorbar(ax6.collections[0], ax=ax6, label='CAL [LSB]')
        
        plt.tight_layout()
        # plt.tight_layout(rect=[0,0,1,0.96])
        
        # Save
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = output_path / f"{chip_names[chip_id].replace('/', '_')}_tdc_analysis_{timestamp}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        
        plt.show()
        plt.close()

def main():
    DATA_DIR = '/home/roy/yf_temp/tamalero/cosmic_run/1009'
    OUTPUT_DIR = '/home/roy/yf_temp/tamalero/cosmic_run/plots/1009'
    
    print("TDC Analysis")
    print("=" * 50)
    
    raw_words = read_dat_files(DATA_DIR)
    if not raw_words:
        return
    
    hits = parse_etroc_data(raw_words)
    if not hits:
        print("No hits found!")
        return
    
    plot_tdc_analysis(hits, OUTPUT_DIR)
    
    print("\n Done!")

if __name__ == "__main__":
    main()
