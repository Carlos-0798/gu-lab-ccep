# Gu Lab CCEP analysis

Analysis scripts for the two ACC-stimulation recordings: CH2 is LP and CH3 is SC.
Each recording contains 30 stimulation commands. CH1 and CH4 are retained in the
saved arrays for inspection but are not used as reference channels.

The repository contains code only. Obtain the original TDT blocks separately.
Keep each block's filenames and accompanying files together.

## Run

Tested with Python 3.12.10 on Windows. From this folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the complete first block in `data/mouse_1` and the second in `data/mouse_2`,
or use `--mouse-1-block` and `--mouse-2-block` to point to their existing folders.
These must be block folders, not their parent tank folder.

```powershell
# LP and SC averages for both mice; bandpass is the default.
.\.venv\Scripts\python.exe filter_ccep.py

# No extra filter / bandpass / notch plus bandpass, side by side.
.\.venv\Scripts\python.exe filter_ccep.py --compare

# Separate 60 Hz notch result, with the earlier -0.1 to 0.2 s display window.
.\.venv\Scripts\python.exe filter_ccep.py --mode notch60 --xmax 0.2

# Artifact-copy comparison and early-transient zoom; builds the control if needed.
.\.venv\Scripts\python.exe artifact_ccep.py
```

For a single recording in another location:

```powershell
.\.venv\Scripts\python.exe filter_ccep.py --mouse 1 --mouse-1-block "D:\TDT recordings\first block"
```

The same block options work for `artifact_ccep.py`. Use `--mouse 2` to run only the
second recording; the other block is not needed. `plot_ccep_averages.py` is the
Mouse 1 shortcut. In PyCharm, open this folder, select `.venv\Scripts\python.exe`
as the interpreter, and run one of these scripts with the same arguments.

Outputs go to `results/filter` and `results/artifact`: PNG figures, mean/SD CSVs,
trial arrays, and processing summaries. Use `--output-dir` for another results
folder, separate from the input blocks. `filter_ccep.py --build-only` saves all
filter plots without opening windows; `artifact_ccep.py --no-show` does the same
for the artifact comparison.

Cached results are checked against inputs, settings, code, package versions, and
NPZ result-file hashes. After changes, use `filter_ccep.py --rebuild` or
`artifact_ccep.py --build`. Switching the selected mouse set in the same output
folder also requires a rebuild. Existing source blocks are never modified.

## Processing

| Item | Implementation |
| --- | --- |
| Red line / time zero | First sample in each group where absolute `eS1r` row 1 exceeds 100; gaps greater than 100 ms separate groups. Stream start times and sample rates map commands to `LFP1` samples. This threshold is specific to these recordings. |
| Trials | Exactly 30 complete epochs, approximately -250 to +500 ms, for each recording. Unexpected counts or incomplete epochs stop processing. |
| Baseline | Subtract each trial's mean over -250 to -50 ms before calculating the mean and SD. |
| Gray region | Mean +/- population standard deviation (`ddof=0`); not SEM or a confidence interval. Faint gray lines are individual trials. |
| Bandpass | First-order Butterworth, 0.5-30.5 Hz, applied forward and backward with `sosfiltfilt`. |
| 60 Hz notch | `iirnotch(60, Q=30)` with `filtfilt`; the combined mode applies the notch before the bandpass. |
| Filtering scope | Full continuous record, each channel separately, before epoch extraction. No filtering across concatenated trials. The boundary-sensitivity check requires at least 4 s of data on each side of every epoch. |
| Display | Default -0.1 to +0.4 s. `--xmax 0.2` restores the shorter view. Filter plots use +/-1 mV by default; `--ylim 0.04` shows +/-40 mV. Clipped traces are identified on the plot. Display limits do not change the saved arrays. |

`no_extra_filter` means no additional offline filter. The saved LFP signals
already contain acquisition-time filtering. Forward-backward filtering can
spread a stimulation transient into surrounding samples, including before zero.

## Artifact comparison and interpretation

`artifact_ccep.py` copies approximately 2 ms immediately before each saved
command into the following 2 ms, on CH2 and CH3, before bandpass filtering.
At 3051.7578125 Hz this is 6 samples (1.96608 ms). The comparison holds the rest
of the processing constant. A separate symmetric short-epoch comparison is saved
in the arrays as a check against the example's epoch-filtering approach.

The main transient in these recordings occurs around 6-7 ms after the command,
outside that replacement window. This test does **not** establish successful
artifact removal. The optional 1 mV rejection rule is counted separately and is
not applied: it would reject all 30 trials in each LP/SC group. All trials remain
in the displayed averages.

The red line is a saved command timestamp, not a measured physical pulse arrival.
Voltage units follow the supplied program's V interpretation; physical amplitude
calibration remains unverified. Early deflections may include stimulation artifact
or amplifier recovery. P1/N1/P2/N2 labels and physiological latencies are therefore
not assigned automatically. Peak identification and any replacement-window change
still need review with Dr. Gu.

## Source and checks

The bandpass settings, plotting style, and pre-event copy method are adapted from
the CCEP example shared by Dr. Bin Gu. Event extraction and channel mapping are
adapted to these recordings. Compared with that example, the main analysis uses
continuous-record filtering, per-trial baseline correction, and the actual command
sample instead of the midpoint of an asymmetric epoch. Baseline is not subtracted
from SD. The original example and laboratory correspondence are not redistributed
here.

Run the synthetic numerical and cache checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These checks do not establish physiological validity or verify physical hardware.
