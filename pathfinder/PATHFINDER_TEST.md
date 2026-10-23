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

Typical runs (D0 H1-H3, 320 Mbps, step-by-step, full records printed).

First run: scan all 256 pixels per chip and save the BL/NW result:

```bash
python pathfinder/pathfinder_test.py --stage qinj --selection '{"0": [1, 2, 3]}' --rate 320 --trigger-delay 471 --step --print-records --blnw all
```

Later runs: load the newest saved full-matrix BL/NW result and skip the scan (nothing new is saved):

```bash
python pathfinder/pathfinder_test.py --stage qinj --selection '{"0": [1, 2, 3]}' --rate 320 --trigger-delay 471 --step --print-records --blnw all --blnw-load latest
```

Defaults used above: `--offset 50`, `--qinj 15` (fC), `--qinj-cnt 1`. `--step` already opens the interactive menu after every step.

Other runs.

Offline preview, no hardware access (default):

```bash
python3 pathfinder/pathfinder_test.py
```

Full 640 Mbps run on daughter card D2: fresh BL/NW scan, one QInj burst, menu afterwards:

```bash
python3 pathfinder/pathfinder_test.py --stage qinj --selection '{"2":"all"}' --rate 640 --interactive
```

Same, but reuse the newest saved full-matrix BL/NW result (skip the scan):

```bash
python3 pathfinder/pathfinder_test.py --stage qinj --selection '{"2":"all"}' --blnw all --blnw-load latest --interactive
```

Pause after every step (menu), so you can tune before the burst is sent in Step 11:

```bash
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
| `--repo-root` | parent of `pathfinder/` | Repository root (where the tamalero drivers are found). |
| `--mapping` | `pathfinder_mapping.yaml` | Mapping file (absolute path, or name relative to `pathfinder/`). |
| `--selection` | `{"0":"all"}` | Daughters/hybrids to read out. |
| `--trigger-selection` | = selection | Daughters/hybrids that contribute to the trigger. |
| `--trigger-logic` | `or` | Combine trigger inputs with `or` / `and`. |
| `--ip`, `--rb-id` | `192.168.0.10`, `0` | KCU IP address and readout-board id. |
| `--board-config` | `default` | Board configuration name. |
| `--rate` | `640` | ETROC link rate, 320 or 640 Mbps. Settings (phase select, trigger delay) are rate-specific. |
| `--trigger-bits` | `1` | Merged trigger bits (0, 1, 2). Written to every selected ETROC (merge on, trigger granularity) and to the FPGA `TRIG_DATA_SIZE` in Step 5.2, and read back. |
| `--trigger-delay` | `469` | FPGA `TRIG_DLY_SEL` (one global delay line). |
| `--l1a-delay` | `501` | Per-ETROC pixel `L1Adelay` (BX). Different from the FPGA delay and from the FC input delay bits. |
| `--scan-selection` | = selection | Subset to phase-scan. |
| `--phase-settings` | `{}` | Fixed phases applied after initialization, e.g. `'{"D3H1":4}'`. |
| `--blnw` / `--pixels` | `sample` | BL/NW coverage: `sample` = 4 pixels/chip (corners), `all` = 256 pixels/chip. |
| `--blnw-load` | off | Skip the BL/NW scan and load saved results: a `.pkl` path or `latest` (section 5). |
| `--no-save` | off | Do not save the BL/NW pickle. |
| `--plot` (`--plots`) / `--no-plots` (`--no-plot`) | no plots | Show BL/NW figures in the workflow. Menu 16 always works. |
| `--offset` | `50` | Threshold DAC = baseline + offset. |
| `--qinj` / `--charge-fc` | `15` | Injected charge in fC (1..32). |
| `--qinj-cnt` / `--qinj-count` | `1` | Injection pulses per burst. |
| `--fifo-reads` | `3` | FIFO read passes after one burst (no extra injections). |
| `--print-records` | off | Also print full header/data/trailer records grouped by E-link. |
| `--fc-target` | none | ETROC for `--stage fc-scan`, e.g. `D3H1`. |
| `--scan-output` | (none) | Override folder for diagnostic JSON files. Default: `pathfinder/results/<kind>/`. |
| `--monitor-seconds` | `3` | Duration of the uplink monitor in the motherboard check. |
| `--wrap` | off | By default long output lines are not wrapped (cut at the terminal edge, so widen the window to see more); `--wrap` restores normal wrapping. Not applied when output is redirected. |
| `--interactive` | off | Open the diagnostic menu after a hardware stage (and at the end). |
| `--step` | off | Pause after every step with the menu, so you can debug and retry in the same session. |

BL/NW results are saved to `pathfinder/results/blnw/blnw_<timestamp>.pkl`.

## 4. Workflow steps

| Step | Title | Action | What to check |
|---|---|---|---|
| 0 | Repository setup | Set repository paths and locate drivers. | Repository and driver paths must match this test installation. |
| 1 | Test parameters | Load control functions and apply command-line selections. | Review mapping, chip selection and connection parameters. |
| 2 | Wiring preview | Resolve selected ETROCs using the mapping YAML. | I2C routes and readout masks should match the connected hardware. |
| 2.1 | Trigger preview | Calculate the trigger mask without writing hardware. | Trigger sources must be included in the readout selection. |
| 3 | KCU connection | Connect to KCU and perform the register loopback test. | Loopback must pass. Initial optical status is not the final post-configuration result. |
| 4 | Motherboard initialization | Create rb, configure declared lpGBTs, configure TRIG clocks and enable the selected data links. | Both declared lpGBT objects must be available; ELINK_WIDTH must read back correctly. |
| 4.1 | Optical-link configuration and checks | Configure optical links, verify readback and monitor uplinks, then check lpGBT status. | For Pathfinder, both READY flags must stay 1 and FEC counters stay 0 at the sampled times. |
| 5 | ETROC initialization | Create the selected ETROC objects. | Selected chips must be connected; motherboard-only runs stop before this step. |
| 5.1 | ETROC communication and register checks | Check I2C communication with the initialized chips, then read the key registers. | A failed I2C check names the ETROC and the error; register readings are printed per chip. |
| 5.2 | Power mode, rate and merge configuration | Apply high power mode, link rate and merged trigger data (ETROC and FPGA trigger bits). | Power-mode, merge and TRIG_DATA_SIZE readbacks must pass; subsequent link retries preserve these settings. |
| 6 | E-link lock check | Check selected E-link locks before pixel calibration. | On failure use compare, masks, phase scan/apply, then retry this check. |
| 6.1 | Link diagnostics | Read settings, masks, controllerState and lock status. | Check optical link, rate/merge, ENABLE/INVERT/TERM and masks before scanning phase. |
| 7 | BL/NW scan settings | Apply pixel coverage and BL/NW save options. | sample scans 4 pixels per chip; all scans 256 pixels per chip. |
| 7.1 | BL/NW scan | Scan selected pixels using the original notebook sequence. | Follow the progress bar and failure summary; saved results use BLNW_OUTPUT_DIR. |
| 7.2 | BL/NW figures | Display maps and applicable histograms unless --no-plots is set. | A plot window may block execution; close it to continue. |
| 8 | Pixel thresholds | Apply baseline-derived DAC thresholds and activate scanned pixels. | Threshold configuration must complete before acquisition. |
| 9 | Self-trigger configuration | Apply the selected trigger mask, combination logic and delay. | Inspect mask, lock and FC diagnostic output. |
| 10 | QInj preparation | Enable injection on calibrated pixels. | No automatic trigger/FC delay tuning is performed. |
| 11 | Data Acquisition | Use current prepared settings. | Errors return to debugging; retry resends an injection burst and resets FIFO. |
| 11.1 | FC input delay scan | Scan all four clk/data delay-bit combinations: 00, 01, 10, 11. | Original FC bits are restored, even on acquisition failure; other chips remain unchanged. |
| 12 | End of test | Mask the declared readout banks using the notebook cleanup step. | This does not power down chips or disable every external trigger source. |

### Run modes

The workflow always runs from Step 0 up to the end point given by `--stage` (section 2). It cannot start in the middle,
because each step needs the objects created by the earlier ones. `--step` and `--interactive` only change when it stops:

| Options | On success | On a failed step |
|---|---|---|
| neither | Runs straight through to the end of the stage, then exits. | Prints the error and traceback, then exits. Hardware settings already written stay in place. |
| `--interactive` | Runs straight through, then opens the menu at the end of the stage. | Opens the menu at the failed step. |
| `--step` | Pauses after every step with the menu (implies `--interactive`). | Opens the menu at the failed step. |

In the menu opened at a step: **r** reruns the step, **Enter/c** continues (only once the step passes), **0** exits. Exiting never powers down hardware.
The menu at the end of the stage has no r / Enter lines.

## 5. Reusing saved BL/NW results

A full-matrix scan is slow. After the first scan, later runs can load it:

```bash
python3 pathfinder/pathfinder_test.py --stage qinj --blnw all --blnw-load latest ...
```

- `latest` picks the newest usable file; unusable files are skipped with a printed reason.
- A saved result must be **complete**, must cover **all selected ETROCs**, and must cover the requested `--blnw` pixels.
  A `sample` (4 pixel) file cannot be used for `--blnw all`; an `all` file can serve `--blnw sample` (only the 4 corner pixels are used).
- Pixels whose scan failed (baseline `None`) are skipped when thresholds are set.
- Step 7.2 / menu 16 plots the loaded data; nothing new is saved.
- Baselines drift with temperature and supply. Rescan after changing the ETROC power supply, slot or conditions.

## 6. Interactive menu

`READ ONLY` = no hardware change; `WRITE HARDWARE` = changes registers/settings.

| # | Action | Kind | What it does | Prompts / notes |
|---|---|---|---|---|
| 0 | Exit | | Leave the menu; hardware is not powered down. | |
| 1 | KCU Status | READ ONLY | Read KCU status, clocks and available E-link lock flags. |  |
| 2 | Uplinks Status | READ ONLY | Read current counters without clearing them. |  |
| 3 | Monitor Uplinks | READ ONLY | Sample READY/FEC once per second for 10 seconds; counters are not reset. |  |
| 4 | Check Clock Registers | READ ONLY | Read ECLK0-27 frequency and drive strength; this does not measure waveforms. |  |
| 5 | Reset FEC Counters | WRITE HARDWARE | Clear historical FEC counts on both uplinks. |  |
| 6 | Configure Optical Links | WRITE HARDWARE | Apply the tested Pathfinder lpGBT/VTRx+ profile, clear FEC counts and verify for 10 seconds. | Only for `pathfinder_mapping.yaml`. |
| 7 | Read Receiver Masks | READ ONLY | Read DAQ/TRIG disable masks without changing them. |  |
| 8 | Enable Selected Readout Links | WRITE HARDWARE | Apply masks for initialized selected ETROCs; other receivers remain disabled. |  |
| 9 | Rerun Bitslip and Check Links | WRITE HARDWARE | Rerun shared bitslip; report selected link status. Does not change masks. | Shared bitslip may realign other channels. |
| 10 | Retry Direct I2C Reads | READ ONLY | Read mapped ETROC register 0 repeatedly, including chips not initialized yet. No reset/default configuration. | Prompts for an I2C selection JSON, e.g. `{"5":"all"}`, and the number of read repetitions (default 3). |
| 11 | Compare Links | READ ONLY | Compare initialized ETROCs side by side; no resets or configuration writes. | Prompts for ETROC IDs (Enter = all) and whether to show only differences. |
| 12 | Check Link Rate | READ ONLY | Compare configured FPGA/lpGBT/ETROC rates. |  |
| 13 | Set Link Rate | WRITE HARDWARE | Set FPGA, lpGBT receiver and ETROC link rate (320/640), then rerun bitslip. Phase select and trigger delay may need to be found again. | Prompts for the link rate: 320 or 640. |
| 14 | Scan EPRX Phase Select | WRITE HARDWARE | Stop acquisition first. Scan selected D/H; restore phases, group modes and readout masks. Shared bitslip may realign other channels. | Prompts for a selection JSON, e.g. `{"3":[1]}`. |
| 15 | Apply EPRX Phase Select | WRITE HARDWARE | Apply explicit phases to selected D/H and retain fixed group mode; restore readout masks afterwards. | Prompts for phases as JSON, e.g. `{"D3H1":4}` (phase 0..14). |
| 16 | Plot BL/NW Results | READ ONLY | Plot current in-memory maps; full scans also show histograms. Close plot windows to return. | Needs BL/NW results in memory. |
| 17 | Scan Trigger Delay | WRITE HARDWARE | Requires prepared QInj. Scan FPGA delay with fixed pixel L1Adelay; restore FPGA delay and readout masks. |  |
| 18 | Set Trigger Delay | WRITE HARDWARE | Set and verify FPGA TRIG_DLY_SEL; pixel L1Adelay is unchanged. | Prompts for the FPGA delay (0..511). |
| 19 | Scan FC Input Delay | WRITE HARDWARE | Scan 00/01/10/11 on one initialized ETROC using QInj; restore original FC bits even on failure. | Prompts for the target ETROC, e.g. `D5H1`. |
| 20 | Set FC Input Delay | WRITE HARDWARE | Set and verify fcClkDelayEn/fcDataDelayEn on explicit targets; keep the requested settings. | Shows the current FC delay bits of all initialized ETROCs first, then prompts for ETROC IDs, e.g. `"D0H0", "D0H1"`, and one value: 00, 01 or 10; shows the table again afterwards. |
| 21 | Charge Injection Acquisition | WRITE HARDWARE | Reuse prepared thresholds/self-trigger. Set charge and pulse count, acquire and save per-link records. | Uses `--qinj` / `--qinj-cnt`; prompts for FIFO read passes and whether to print full records. |
| 22 | Read ADC Monitoring | READ ONLY | Configure mapped ADC channels, calibrate and read values. Missing daughter boards may show out-of-range readings. |  |

Menus 17, 19 and 21 appear only after QInj preparation (`--stage qinj` / `fc-scan`).
They enable the selected links temporarily, so they also work after Step 12 has masked the links.

## 7. Per-slot settings in `pathfinder_mapping.yaml`

Settings are rate- and slot-specific. At 640 Mbps expect to need a per-slot `TRIG_DLY_SEL` and lpGBT `PHASESELECT`;
320 Mbps worked with the defaults. A hybrid that needs the FC data delay (found on D2H0 at 640 Mbps) is listed in the mapping:

```yaml
fc_delay:
  clk_delay_enabled: ["D1H0"]           # fcClkDelayEn = 1
  data_delay_enabled: ["D0H0", "D2H0"]  # fcDataDelayEn = 1
```

Both keys are optional. An ETROC listed in either list gets both bits written during initialization:
each bit is 1 if the ETROC is in the matching list, 0 otherwise (so `D0H0` above gets clk=0, data=1; `D2H0` listed in both gets clk=1, data=1).
ETROCs in neither list are not touched. The bits only act in manual FC alignment mode (`fcSelfAlignEn=0`).
This needs the `configure_fc_delay` function in `pathfinder_control.py`.

## 8. If something goes wrong

General: read the `[FAILED] <step>` line, fix the cause, press **r** to rerun the step.
Use `--step --interactive` to stay in one session. Partial hardware configuration may remain after a failure.
Power cycling the ETROC supply resets settings such as `fcDataDelayEn` to 0.

| Symptom | What to check / do |
|---|---|
| `InitializationError` during **FC termination** | The FC termination step needs all four ETROCs of a daughter card to answer on I2C. Check the others (menu 10). |
| I2C read of a hybrid fails (menu 10; register 0 should read `0x2C` = 44) | The four hybrids of a daughter share one I2C bus (addresses 0x60-0x63), so one bad chip can break the others. Check the ETROC power supply, reseat the daughter card, power cycle. A supply change fixed D5 once. |
| One hybrid does not lock at 640 Mbps (Step 6 / menu 11) | Rate-specific. Scan EPRX phase select (menu 14) and apply the result (menu 15 or `--phase-settings`); the lpGBT EQ setting (`EPRX{e}EQ`) can also help. EPRX termination is enabled in the lpGBT init. |
| Links lock but one chip is missing or has too few events | Read the per-E-link table. Event E-link number = 28 x bank + lpGBT E-link (TRIG E4/E6/E8/E10 = 32/34/36/38). No headers: link or trigger path. Headers present but the L1 counter is far below the pulse count: L1A/FC not received, try menu 19/20 (`fcDataDelayEn`). Wrong or empty BCID window: scan the trigger delay (menu 17/18) or change `--l1a-delay`. |
| `invalidFCCount` constant but L1As are lost | It only counts malformed FC commands. Judge by the received L1A count and try the FC delay bits (menu 19/20). |
| `controllerState` is not 11 (`ETROC.is_good()` fails) | Check the ETROC supply. Seen once as 7 on one hybrid; cause not understood. |
| Trigger does not reach all chips | Trigger mask bit = 14 x bank + E-link/2; TRIG E4..E10 are bits 16-19 (`0xF0000`). Check `--trigger-selection`. |
| "Link lock lost during QInj preparation" | The script refuses to send the burst. Rerun bitslip (menu 9), check links (menu 11), then retry (r). |
| `--blnw-load`: "No usable saved BL/NW result" | The file is incomplete, was a `sample` scan while `all` is requested, or lacks one of the selected ETROCs. Rescan once without `--blnw-load`. |
| KCU does not answer / uhal errors | Check `--ip` (default 192.168.0.10), power and the Ethernet link. |
| Start over | Press r for the failed step. For a clean start rerun the script from the beginning (exiting does not reset the hardware). |

## 9. Notes

- Menu 21 and `--print-records` print the full header/data/trailer records per E-link (the notebook-style output).
- Nothing touches hardware unless `--stage` is `motherboard` or later.
- Output files: BL/NW pickles in `pathfinder/results/blnw/`, diagnostic JSON files in `pathfinder/results/trigger_delay/`, `qinj/`, `fc_delay/`, `phase_scan/` (or `--scan-output` if given).
