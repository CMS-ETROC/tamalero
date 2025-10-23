import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tamalero.FIFO import merge_words

def read_dat_files(data_dir):
    data_path = Path(data_dir)
    dat_files = sorted(data_path.glob('file_*_CE.dat'))
    
    if not dat_files:
        print(f"No .dat files found in {data_dir}")
        return []
    
    print(f"Found {len(dat_files)} .dat files:")
    for f in dat_files:
        print(f"  - {f.name}")
    
    all_words = []
    for dat_file in dat_files:
        with open(dat_file, 'rb') as f:
            data = f.read()
        num_words = len(data) // 4
        words = struct.unpack(f'<{num_words}I', data)
        all_words.extend(words)
    
    print(f"Total 32-bit words: {len(all_words)}")
    return all_words

from tamalero.DataFrame import DataFrame

def parse_etroc_data(raw_words):

    df = DataFrame('ETROC2')
    merged_64bit = merge_words(raw_words)
    print(f"Merged into {len(merged_64bit)} 64-bit words")
    
    hits = []
    header_count = data_count = trailer_count = filler_count = 0
    
    for word in merged_64bit:
        data_type, parsed_data = df.read(word, quiet=True)
        
        if data_type == 'header':
            header_count += 1
        elif data_type == 'data':
            data_count += 1
            elink = parsed_data.get('elink', None)
            row_id = parsed_data.get('row_id', None)
            col_id = parsed_data.get('col_id', None)
            
            if elink is not None and row_id is not None and col_id is not None:
                if 0 <= row_id < 16 and 0 <= col_id < 16:
                    hits.append((int(elink), int(row_id), int(col_id)))
        elif data_type == 'trailer':
            trailer_count += 1
        elif data_type == 'filler':
            filler_count += 1
    
    print(f"Headers: {header_count}, Data: {data_count}, Trailers: {trailer_count}, Fillers: {filler_count}")
    print(f"Valid hits: {len(hits)}")
    return hits

def create_hitmap(hits):
    """Create hit maps per chip"""
    elink_to_chip = {0: 0, 4: 1, 8: 2, 12: 3}
    
    active_elinks = set(hit[0] for hit in hits)
    print(f"Active E-links: {sorted(active_elinks)}")
    
    hit_maps = {}
    for elink in active_elinks:
        if elink in elink_to_chip:
            chip_id = elink_to_chip[elink]
            hit_maps[chip_id] = np.zeros((16, 16), dtype=int)
    
    for elink, row, col in hits:
        if elink in elink_to_chip:
            chip_id = elink_to_chip[elink]
            if chip_id in hit_maps:
                hit_maps[chip_id][row, col] += 1
    
    return hit_maps

def plot_hitmap(hit_maps, output_dir):
    """Plot pixel hit maps"""
    num_chips = len(hit_maps)
    if num_chips == 0:
        print("No data to plot!")
        return

    if num_chips == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 9))
        axes = [axes]
    elif num_chips == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
    
    elink_map = {0: 0, 1: 4, 2: 8, 3: 12}
    plot_idx = 0
    
    for chip_id in sorted(hit_maps.keys()):
        ax = axes[plot_idx]
        hit_map = hit_maps[chip_id]
        
        im = ax.imshow(hit_map, cmap='viridis', aspect='auto', origin='lower')

        for row in range(16):
            for col in range(16):
                count = int(hit_map[row, col])
                text_color = 'white' if count < hit_map.max() * 0.6 else 'black'
                ax.text(col, row, f'{count}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=8)
        
        ax.set_xlabel('Col', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        elink = elink_map.get(chip_id, chip_id*4)
        ax.set_title(f'Chip {chip_id+1} (E-link {elink})', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Hit Count', rotation=270, labelpad=20)

        total = int(hit_map.sum())
        max_hits = int(hit_map.max())
        mean_hits = hit_map.mean()
        print(f"\nChip {chip_id+1}:")
        print(f"  Total hits: {total}")
        print(f"  Max hits: {max_hits}")
        print(f"  Mean hits: {mean_hits:.2f}")
        
        plot_idx += 1
    

    if num_chips < len(axes):
        for idx in range(num_chips, len(axes)):
            axes[idx].set_visible(False)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f'ETROC Pixel Hit Map | {timestamp}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    save_path = output_path / f"hitmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    
    plt.show()

def main():
    DATA_DIR = '/home/roy/yf_temp/tamalero/cosmic_run/10-17_14-16-59'
    OUTPUT_DIR = '/home/roy/yf_temp/tamalero/cosmic_run/plots/10-20'
    
    print("ETROC Pixel Hit Map Plotter")
    print("=" * 50)
    
    raw_words = read_dat_files(DATA_DIR)
    if not raw_words:
        return
    
    hits = parse_etroc_data(raw_words)
    if not hits:
        print("No hits found!")
        return
    
    hit_maps = create_hitmap(hits)
    plot_hitmap(hit_maps, OUTPUT_DIR)
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()