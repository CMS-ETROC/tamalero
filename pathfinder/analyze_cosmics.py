#!/usr/bin/env python3
"""Quick look at a cosmics run folder (written by pathfinder_test.py --stage cosmics).

For every file_*_CE.dat it decodes the ETROC2 words, counts the hits per pixel for every E-link and
draws 16x16 maps with (15, 15) at the top-left and (0, 0) at the bottom-right (the physical numbering).
If the run used --inject-mode pattern, it also tells which injection pattern each file looks like
(and whether a left-right or up-down mirrored pattern fits better).

Run it on the computer where the data are (nothing has to be copied):

    python pathfinder/analyze_cosmics.py results/cosmics/<UTC time>

It writes cosmics_maps.png and cosmics_summary.txt into the run folder. Send those two small files
(or just paste the summary) instead of the .dat files.

Options: --elink N (only this E-link), --max-pairs N (decode at most N hits per file, default: all).
"""
import argparse
import ast
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def load_patterns():
    """The injection patterns, taken from pathfinder_test.py without importing it (it needs the hardware drivers)."""
    source = (HERE / "pathfinder_test.py").read_text(encoding="utf-8")
    wanted = {"FONT_4X7", "FONT_4X6", "text_pixels", "make_inject_patterns"}
    parts = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            parts.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets):
            parts.append(ast.get_source_segment(source, node))
    namespace = {}
    exec("\n\n".join(parts), namespace)
    return {name: set(pixels) for name, pixels in namespace["make_inject_patterns"]()}


def decode_file(path, max_pairs):
    """Vectorised decoding with the masks of configs/dataformat.yaml (fast enough for files with millions of hits)."""
    import yaml
    fmt = yaml.safe_load(open(REPO / "configs" / "dataformat.yaml"))["ETROC2"]
    ident, data_fields = fmt["identifiers"], fmt["data"]["data"]

    def field(values, name):
        return ((values & np.uint64(data_fields[name]["mask"])) >> np.uint64(data_fields[name]["shift"])).astype(np.int64)

    def is_kind(values, name):
        return (values & np.uint64(ident[name]["mask"])) == np.uint64(ident[name]["frame"])

    words = np.fromfile(path, dtype="<u4").astype(np.uint64)
    # One hit = two words (low, high); the high word is below 2**24. Find which parity fits.
    best = None
    for start in (0, 1):
        high = words[start + 1::2]
        n = min(len(words[start::2]), len(high))
        score = float((high[:n] >> np.uint64(24) == 0).mean()) if n else 0.0
        if best is None or score > best[1]:
            best = (start, score)
    start = best[0]
    low, high = words[start::2], words[start + 1::2]
    n = min(len(low), len(high), max_pairs)
    merged = low[:n] | (high[:n] << np.uint64(32))
    header = is_kind(merged, "header")
    filler = is_kind(merged, "filler")
    data = ~header & is_kind(merged, "data")
    trailer = ~header & ~data & ~filler & is_kind(merged, "trailer")
    kinds = {"header": int(header.sum()), "data": int(data.sum()), "trailer": int(trailer.sum()), "filler": int(filler.sum())}
    row, col, elink = field(merged[data], "row_id"), field(merged[data], "col_id"), field(merged[data], "elink")
    ok = (row < 16) & (col < 16)
    counts = {}
    for link in np.unique(elink):
        picked = ok & (elink == link)
        matrix = np.zeros((16, 16), int)
        np.add.at(matrix, (row[picked], col[picked]), 1)
        counts[int(link)] = matrix
    # Pixels seen in each window of 100000 words; used to find short bursts (an injection) between hot pixels.
    position = np.flatnonzero(data)[ok]
    windows = {}
    for link in counts:
        picked = ok & (elink == link)
        win = position[picked[ok]] // 100000
        for w in np.unique(win):
            sel = win == w
            windows.setdefault(link, {})[int(w)] = set(zip(row[picked][sel].tolist(), col[picked][sel].tolist()))
    return {"words": len(words), "pairs": n, "alignment": start, "kinds": kinds, "counts": counts, "windows": windows}


def compare_with_patterns(hit_pixels, patterns):
    """Overlap of the pixels that fired with every pattern, also for the mirrored versions."""
    rows = []
    for name, pattern in patterns.items():
        for label, transform in (("as drawn", lambda r, c: (r, c)),
                                 ("left-right mirrored", lambda r, c: (r, 15 - c)),
                                 ("up-down mirrored", lambda r, c: (15 - r, c))):
            pixels = {transform(r, c) for r, c in pattern}
            found = len(pixels & hit_pixels)
            extra = len(hit_pixels - pixels)
            rows.append((found / len(pixels), name, label, found, len(pixels), extra))
    rows.sort(reverse=True)
    return rows[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder")
    parser.add_argument("--elink", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=10**9)
    args = parser.parse_args()

    folder = Path(args.run_folder)
    files = sorted(folder.glob("file_*_CE.dat"), key=lambda p: int(p.name.split("_")[1]))
    if not files:
        sys.exit(f"No file_*_CE.dat in {folder}")
    info = json.loads((folder / "run_info.json").read_text()) if (folder / "run_info.json").exists() else {}
    patterns = load_patterns()
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    say(f"Run folder: {folder}")
    if info:
        say(f"status: {info.get('status')}; injection: {info.get('injection')}")
    injections = folder / "injections.csv"
    if injections.exists():
        say("injections.csv:")
        for line in injections.read_text().splitlines():
            say("  " + line)

    panels = []
    for path in files:
        result = decode_file(path, args.max_pairs)
        say("")
        say(f"{path.name}: {result['words']} words, {result['pairs']} words-pairs decoded (parity {result['alignment']}), kinds {result['kinds']}")
        for elink, counts in sorted(result["counts"].items()):
            if args.elink is not None and elink != args.elink:
                continue
            fired = {(int(r), int(c)) for r, c in np.argwhere(counts > 0)}
            say(f"  E-link {elink}: {int(counts.sum())} hits, {len(fired)} different pixels, max {int(counts.max())} hits in one pixel")
            for score, name, label, found, size, extra in compare_with_patterns(fired, patterns):
                say(f"      pattern {name:7s} {label:20s}: {found}/{size} pattern pixels fired ({score:.0%}), {extra} other pixels fired")
            panels.append((path.name, elink, counts))
            hot = {(int(r), int(c)) for r, c in np.argwhere(counts > 0.5 * counts.max())} if counts.max() > 1000 else set()
            if hot:
                say(f"      always-firing pixels (more than half of the maximum count): {sorted(hot)}")
            seen = collections.Counter()
            for w, pixels in result["windows"].get(elink, {}).items():
                extra = pixels - hot
                if extra:
                    seen[w] = extra
            for w, extra in sorted(seen.items())[:6]:
                say(f"      window {w} (words {w * 100000}..{(w + 1) * 100000}): {len(extra)} pixels besides the always-firing ones")
                for score, name, label, found, size, other in compare_with_patterns(extra, patterns)[:1]:
                    say(f"          best pattern {name} {label}: {found}/{size} ({score:.0%}), {other} other")

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = panels[:12]
        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (name, elink, counts) in zip(axes.flat, panels):
            axis.axis("on")
            shown = counts[::-1, ::-1]       # row 15 at the top, column 15 at the left: (0, 0) bottom-right
            axis.imshow(np.where(shown > 0, shown, np.nan), cmap="viridis", interpolation="nearest")
            axis.set_title(f"{name}  E-link {elink}", fontsize=8)
            axis.set_xticks(range(16)); axis.set_xticklabels([15 - j for j in range(16)], fontsize=5)
            axis.set_yticks(range(16)); axis.set_yticklabels([15 - i for i in range(16)], fontsize=5)
            axis.set_xticks(np.arange(-.5, 16), minor=True); axis.set_yticks(np.arange(-.5, 16), minor=True)
            axis.grid(which="minor", color="lightgray", linewidth=0.3); axis.tick_params(which="minor", length=0)
        figure.suptitle("Hits per pixel; (15,15) top-left, (0,0) bottom-right", fontsize=9)
        figure.tight_layout()
        figure.savefig(folder / "cosmics_maps.png", dpi=130)
        say("")
        say(f"Wrote {folder / 'cosmics_maps.png'}")
    (folder / "cosmics_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {folder / 'cosmics_summary.txt'}")


if __name__ == "__main__":
    main()
