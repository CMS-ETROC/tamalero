# Pathfinder test script (`pathfinder_test.py`)

Staged bring-up and charge-injection test for the ETROC2 Pathfinder system
(KCU105 + lpGBT motherboard, DAQ lpGBT #1, TRIG lpGBT #2, daughter cards D0-D6, hybrids H0-H3).
It follows `pathfinder_test.ipynb` (sections 0-12) and uses `pathfinder_control.py`
and `pathfinder_mapping.yaml`.

> Status: the workflow has been used on the bench with short FFC cables. Behavior with long cables is untested.

## 1. Run it

Place `pathfinder_test.py` in the repository's `pathfinder/` directory and run it from the repo:

```bash
python3 pathfinder/pathfinder_test.py --stage <stage> [options]
```

Typical runs (D0 H1-H3, 320 Mbps, step-by-step, full records printed):

```bash
# 1) First run: scan all 256 pixels per chip and save the BL/NW result
python pathfinder/pathfinder_test.py --stage qinj --selection '{"0": [1, 2, 3]}' --rate 320 --trigger-delay 471 --step --print-records --blnw all

# 2) Later runs: load the newest saved full-matrix BL/NW result and skip the scan (nothing new is saved)
python pathfinder/pathfinder_test.py --stage qinj --selection '{"0": [1, 2, 3]}' --rate 320 --trigger-delay 471 --step --print-records --blnw all --blnw-load latest
```

Defaults used above: `--offset 50`, `--qinj 15` (fC), `--qinj-cnt 1`. `--step` already opens the interactive menu after every step.

Other runs:

```bash
# Offline preview, no hardware access (default)
python3 pathfinder/pathfinder_test.py

# Full 640 Mbps run on daughter card D2: fresh BL/NW scan, one QInj burst, menu afterwards
python3 pathfinder/pathfinder_test.py --stage qinj --selection '{"2":"all"}' --rate 640 --interactive

# Same, but reuse the newest saved full-matrix BL/NW result (skip the scan)
python3 pathfinder/pathfinder_test.py --stage qinj --selection '{"2":"all"}' --blnw all --blnw-load latest --interactive

# Pause after every step (menu), so you can tune before the burst is sent in Step 11
python3 pathfinder/pathfinder_test.py --stage qinj --selection '{"2":"all"}' --step --interactive
```

`--selection` is JSON `{daughter: "all" | [hybrids]}`, e.g. `'{"1":"all","3":[1]}'` = all of D1 and H1 of D3. Default: `{"0":"all"}`.

## 2. Stages (`--stage`)

Each stage runs all earlier ones first.

| Stage | What it does |
|---|---|
| `preview` (default) | Prints the run plan only. No hardware access. |
| `motherboard` | Configures KCU/lpGBT/clocks and checks the motherboard. Stops before ETROC initialization. |
| `links` | Initializes ETROCs, sets rate/merge options and runs the E-link lock check (Step 6). |
| `phase-scan` | `links` plus an lpGBT phase-select scan on `--scan-selection` (default: `--selection`). |
| `blnw` | Through the BL/NW baseline / noise-width scan (Step 7); stops after the figures. |
| `qinj` | Everything: thresholds, self-trigger, QInj preparation, **sends the injection burst**, reads the FIFO, saves results, then masks the links (Step 12). |
| `fc-scan` | Like `qinj`, but Step 11 scans the FC input delay on `--fc-target` (required). |

## 3. Options

| Option | Default | Meaning |
|---|---|---|
| `--mapping` | `pathfinder_mapping.yaml` | Mapping file (absolute path, or name relative to `pathfinder/`). |
| `--selection` | `{"0":"all"}` | Daughters/hybrids to read out. |
| `--trigger-selection` | = selection | Daughters/hybrids that contribute to the trigger. |
| `--trigger-logic` | `or` | Combine trigger inputs with `or` / `and`. |
| `--ip`, `--rb-id` | `192.168.0.10`, `0` | KCU IP address and readout-board id. |
| `--board-config` | `default` | Board configuration name. |
| `--rate` | `640` | ETROC link rate, 320 or 640 Mbps. Settings (phase select, trigger delay) are rate-specific. |
| `--trigger-bits` | `1` | Merged trigger bits (0, 1, 2). |
| `--trigger-delay` | `469` | FPGA `TRIG_DLY_SEL` (one global delay line). |
| `--l1a-delay` | `501` | Per-ETROC pixel `L1Adelay` (BX). Different from the FPGA delay and from the FC input delay bits. |
| `--scan-selection` | = selection | Subset to phase-scan. |
| `--phase-settings` | `{}` | Fixed phases applied after initialization, e.g. `'{"D3H1":4}'`. |
| `--blnw` / `--pixels` | `sample` | BL/NW coverage: `sample` = 4 pixels/chip (corners), `all` = 256 pixels/chip. |
| `--blnw-load` | off | Skip the BL/NW scan and load saved results: a `.pkl` path or `latest` (section 5). |
| `--no-save` | off | Do not save the BL/NW pickle. |
| `--plot` / `--no-plots` | no plots | Show BL/NW figures in the workflow. Menu 22 always works. |
| `--offset` | `50` | Threshold DAC = baseline + offset. |
| `--qinj` / `--charge-fc` | `15` | Injected charge in fC (1..32). |
| `--qinj-cnt` / `--qinj-count` | `1` | Injection pulses per burst. |
| `--fifo-reads` | `3` | FIFO read passes after one burst (no extra injections). |
| `--print-records` | off | Also print full header/data/trailer records grouped by E-link. |
| `--fc-target` | none | ETROC for `--stage fc-scan`, e.g. `D3H1`. |
| `--scan-output` | `diagnostic_results` | Folder for diagnostic JSON files. |
| `--monitor-seconds` | `3` | Duration of the uplink monitor in the motherboard check. |
| `--interactive` | off | Open the diagnostic menu after a hardware stage (and at the end). |
| `--step` | off | Pause after every step with the menu, so you can debug and retry in the same session. |

BL/NW results are saved to `pathfinder/results/blnw/blnw_<timestamp>.pkl`.

## 4. Workflow steps

| Step | Content |
|---|---|
| 0-4 | Setup, KCU/lpGBT/clock configuration, optical links, readout links enabled for the selected ETROCs. |
| 5 | ETROC communication and register checks (5.1), power mode (5.2), rate and merge configuration (5.3). |
| 6 | E-link lock check. 6.1 = link diagnostics when it fails. |
| 7 | BL/NW settings, scan (7.1), figures (7.2). |
| 8 | Pixel thresholds (baseline + offset). |
| 9 | Self-trigger configuration and FC status check (prints `fcClkDelayEn` / `fcDataDelayEn`). |
| 10 | QInj preparation. |
| 11 | Data acquisition (sends the burst for `--stage qinj`). 11.1 = FC input delay scan. |
| 12 | End of test: readout banks are masked (chips are not powered down). |

On a failed step the script stops (with `--interactive` / `--step` it opens the menu instead).
In that menu: **R** reruns the step, **Enter/c** continues (only once the step passes), **0** exits. Exiting never powers down hardware.

## 5. Reusing saved BL/NW results

A full-matrix scan is slow. After the first scan, later runs can load it:

```bash
python3 pathfinder/pathfinder_test.py --stage qinj --blnw all --blnw-load latest ...
```

- `latest` picks the newest usable file; unusable files are skipped with a printed reason.
- A saved result must be **complete**, must cover **all selected ETROCs**, and must cover the requested `--blnw` pixels.
  A `sample` (4 pixel) file cannot be used for `--blnw all`; an `all` file can serve `--blnw sample` (only the 4 corner pixels are used).
- Pixels whose scan failed (baseline `None`) are skipped when thresholds are set.
- Step 7.2 / menu 22 plots the loaded data; nothing new is saved.
- Baselines drift with temperature and supply. Rescan after changing the ETROC power supply, slot or conditions.

## 6. Interactive menu

`READ ONLY` = no hardware change; `WRITE HARDWARE` = changes registers/settings.

| # | Action | Notes |
|---|---|---|
| 1 | KCU status | |
| 2 | Uplink READY/FEC | Counters are not cleared. A saturated 65535 proves nothing. |
| 3 | Monitor uplinks 10 s | |
| 4 | Clock register readback | Register values, not waveforms. |
| 5 | Reset FEC counters | |
| 6 | Configure and verify optical links | Only for `pathfinder_mapping.yaml`. |
| 7 | Compare ETROC links | Side-by-side status of initialized ETROCs. |
| 8 | Scan phase select | JSON selection, e.g. `{"3":[1]}`. Stop acquisition first. |
| 9 | Apply phase settings | JSON `{"D3H1":4}`; phase 0..14. |
| 10 | Scan trigger delay | Needs prepared QInj. Restores the FPGA delay afterwards. |
| 11 | Set trigger delay | Sets `TRIG_DLY_SEL`; `L1Adelay` unchanged. |
| 12 | Charge injection acquisition | Uses `--qinj` / `--qinj-cnt`; asks for FIFO read passes and whether to print full records. |
| 13 | Scan FC input delay | One ETROC; tries clk/data = 00, 01, 10, 11 with QInj; restores the original bits. |
| 14 | Set FC input delay | One value (00/01/10) for one or more ETROCs, e.g. `"D0H0", "D0H1"`. Kept after return. |
| 15 | Retry direct I2C reads | Reads register 0 of mapped ETROCs, also before initialization. |
| 16 | Rerun bitslip and check links | Shared bitslip may realign other channels. |
| 17 | Read receiver masks | |
| 18 | Check link rates | FPGA / lpGBT / ETROC rates. |
| 19 | Enable selected readout links | Other receivers stay masked. |
| 20 | Reset one global readout | Stop triggers first; reads `controllerState` afterwards. |
| 21 | Read ADC monitoring | Configures the mapped ADC channels, calibrates, reads. Missing boards may read out of range. |
| 22 | Plot BL/NW results | Needs BL/NW results in memory. |

Menus 10, 12 and 13 appear only after QInj preparation (`--stage qinj` / `fc-scan`).
They enable the selected links temporarily, so they also work after Step 12 has masked the links.

## 7. Per-slot settings in `pathfinder_mapping.yaml`

Settings are rate- and slot-specific. At 640 Mbps expect to need a per-slot `TRIG_DLY_SEL` and lpGBT `PHASESELECT`;
320 Mbps worked with the defaults. A hybrid that needs the FC data delay (found on D2H0 at 640 Mbps) is listed in the mapping:

```yaml
fc_delay:
  data_delay_enabled: ["D2H0"]
```

Listed ETROCs get `fcClkDelayEn=0`, `fcDataDelayEn=1` during initialization (requires manual FC alignment, `fcSelfAlignEn=0`).
This needs the `configure_fc_delay` function in `pathfinder_control.py`.

## 8. If something goes wrong

General: read the `[FAILED] <step>` line, fix the cause, press **R** to rerun the step.
Use `--step --interactive` to stay in one session. Partial hardware configuration may remain after a failure.
Power cycling the ETROC supply resets settings such as `fcDataDelayEn` to 0.

| Symptom | What to check / do |
|---|---|
| `InitializationError` during **FC termination** | The FC termination step needs all four ETROCs of a daughter card to answer on I2C. Check the others (menu 15). |
| I2C read of a hybrid fails (menu 15; register 0 should read `0x2C` = 44) | The four hybrids of a daughter share one I2C bus (addresses 0x60-0x63), so one bad chip can break the others. Check the ETROC power supply, reseat the daughter card, power cycle. A supply change fixed D5 once. |
| One hybrid does not lock at 640 Mbps (Step 6 / menu 7) | Rate-specific. Scan phase select (menu 8) and apply the result (menu 9 or `--phase-settings`); the lpGBT EQ setting (`EPRX{e}EQ`) can also help. EPRX termination is enabled in the lpGBT init. |
| Links lock but one chip is missing or has too few events | Read the per-E-link table. Event E-link number = 28 x bank + lpGBT E-link (TRIG E4/E6/E8/E10 = 32/34/36/38). No headers: link or trigger path. Headers present but the L1 counter is far below the pulse count: L1A/FC not received, try menu 13/14 (`fcDataDelayEn`). Wrong or empty BCID window: scan the trigger delay (menu 10/11) or change `--l1a-delay`. |
| `invalidFCCount` constant but L1As are lost | It only counts malformed FC commands. Judge by the received L1A count and try the FC delay bits (menu 13/14). |
| `controllerState` is not 11 (`ETROC.is_good()` fails) | Try menu 20 (reset one global readout) and check the supply. Seen once as 7 on one hybrid; cause not understood. |
| Trigger does not reach all chips | Trigger mask bit = 14 x bank + E-link/2; TRIG E4..E10 are bits 16-19 (`0xF0000`). Check `--trigger-selection`. |
| "Link lock lost during QInj preparation" | The script refuses to send the burst. Rerun bitslip (menu 16), check links (menu 7), then retry (R). |
| `--blnw-load`: "No usable saved BL/NW result" | The file is incomplete, was a `sample` scan while `all` is requested, or lacks one of the selected ETROCs. Rescan once without `--blnw-load`. |
| KCU does not answer / uhal errors | Check `--ip` (default 192.168.0.10), power and the Ethernet link. |
| Start over | Press R for the failed step. For a clean start rerun the script from the beginning (exiting does not reset the hardware). |

## 9. Notes

- Menu 12 and `--print-records` print the full header/data/trailer records per E-link (the notebook-style output).
- Nothing touches hardware unless `--stage` is `motherboard` or later.
- Output files: BL/NW pickles in `pathfinder/results/blnw/`, diagnostics as JSON in `--scan-output`.
