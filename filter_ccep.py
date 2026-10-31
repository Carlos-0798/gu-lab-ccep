"""Average command-aligned CCEP trials and compare offline filter settings."""

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import scipy
import tdt
from scipy.signal import butter, filtfilt, freqz, freqz_sos, iirnotch, sosfiltfilt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/filter"
BLOCKS = {
    1: ROOT / "data/mouse_1",
    2: ROOT / "data/mouse_2",
}
SITES = {2: "LP", 3: "SC"}
MODES = ("no_extra_filter", "bandpass", "notch60", "notch60_bandpass")
LABELS = {
    "no_extra_filter": "No extra filter",
    "bandpass": "BP 0.5-30.5 Hz",
    "notch60": "Notch 60 Hz",
    "notch60_bandpass": "Notch 60 Hz + BP 0.5-30.5 Hz",
}
CONFIG = {
    "pipeline_version": 2,
    "command_threshold": 100.0,
    "command_group_gap_s": 0.1,
    "expected_trials": 30,
    "bandpass_Hz": [0.5, 30.5],
    "butter_order_parameter": 1,
    "bandpass_function": "sosfiltfilt; axis=-1; odd padding; scipy default padlen",
    "notch_Hz": 60.0,
    "notch_Q": 30.0,
    "notch_function": "iirnotch then filtfilt; axis=-1; scipy defaults",
    "combined_order": "notch60 then bandpass",
    "filter_scope": "Full continuous LFP1 record before epoch extraction",
    "baseline_s": [-0.25, -0.05],
    "epoch_s": [-0.25, 0.5],
    "sd_ddof": 0,
    "artifact_replacement": None,
    "trial_rejection": None,
    "clipping": None,
    "time_zero": "Saved eS1r command, not calibrated physical pulse arrival",
    "units": "Input V interpretation retained from supplied program; calibration unverified",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_hashes():
    hashes = {}
    for mouse, block in BLOCKS.items():
        if (
            not block.is_dir()
            or not list(block.glob("*.tsq"))
            or not list(block.glob("*.tev"))
        ):
            raise FileNotFoundError(
                f"Mouse {mouse}: expected a TDT block with TSQ/TEV files at {block}. Use --mouse-{mouse}-block."
            )
        for path in sorted(p for p in block.rglob("*") if p.is_file()):
            hashes[f"mouse_{mouse}/{path.relative_to(block).as_posix()}"] = digest(path)
    return hashes


def runtime_signature():
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tdt": tdt.__version__,
        "analysis_sha256": digest(Path(__file__)),
    }


def add_path_arguments(parser):
    parser.add_argument("--mouse-1-block", type=Path, default=ROOT / "data/mouse_1")
    parser.add_argument("--mouse-2-block", type=Path, default=ROOT / "data/mouse_2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")


def configure_paths(args, mice):
    global OUT, BLOCKS
    BLOCKS = {mouse: getattr(args, f"mouse_{mouse}_block").resolve() for mouse in mice}
    output = args.output_dir.resolve()
    for block in BLOCKS.values():
        if output.is_relative_to(block) or block.is_relative_to(output):
            raise ValueError("Input blocks and output directory must be separate.")
    OUT = output / "filter"


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def coefficients(fs):
    if not np.isfinite(fs) or fs <= 120:
        raise ValueError("Sampling frequency does not support these filter settings")
    return butter(1, [0.5, 30.5], btype="bandpass", output="sos", fs=fs), iirnotch(
        60, 30, fs=fs
    )


def apply_filter(values, fs, mode):
    if mode not in MODES:
        raise ValueError(mode)
    y = np.array(values, dtype=np.float64, copy=True)
    sos, (b, a) = coefficients(fs)
    if mode in ("notch60", "notch60_bandpass"):
        y = filtfilt(b, a, y, axis=-1)
    if mode in ("bandpass", "notch60_bandpass"):
        y = sosfiltfilt(sos, y, axis=-1)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite filter output")
    return y


def command_centers(block):
    lfp, cmd = block.streams["LFP1"], block.streams["eS1r"]
    command = np.atleast_2d(cmd.data)[0].astype(float)
    if not np.isfinite(command).all():
        raise ValueError("Nonfinite stimulation command samples")
    hits = np.flatnonzero(np.abs(command) > CONFIG["command_threshold"])
    if not hits.size:
        raise ValueError("No eS1r command events exceed the configured threshold")
    groups = [
        g
        for g in np.split(
            hits,
            np.flatnonzero(np.diff(hits) > int(CONFIG["command_group_gap_s"] * cmd.fs))
            + 1,
        )
        if len(g)
    ]
    onsets = np.array([g[0] / float(cmd.fs) + float(cmd.start_time) for g in groups])
    centers = np.rint((onsets - float(lfp.start_time)) * float(lfp.fs)).astype(int)
    if len(centers) != CONFIG["expected_trials"]:
        raise ValueError(
            f"Expected {CONFIG['expected_trials']} events, got {len(centers)}"
        )
    return onsets, centers


def extract(values, centers, offsets):
    if np.any(centers + offsets[0] < 0) or np.any(
        centers + offsets[-1] >= values.shape[-1]
    ):
        raise ValueError("Incomplete trial: refuse silent exclusion")
    return np.stack([values[:, center + offsets] for center in centers]) * 1e6


def baseline_correct(epochs, time):
    mask = (time >= -0.25) & (time < -0.05)
    if not mask.any():
        raise ValueError("Epoch does not contain the baseline window")
    baselines = epochs[..., mask].mean(axis=-1, keepdims=True)
    return epochs - baselines, baselines


def sine_amplitude(time, signal, frequency=60):
    design = np.column_stack(
        (
            np.sin(2 * np.pi * frequency * time),
            np.cos(2 * np.pi * frequency * time),
            np.ones(len(time)),
            time,
        )
    )
    fitted = np.linalg.lstsq(design, signal, rcond=None)[0]
    return float(np.hypot(fitted[0], fitted[1]))


def build():
    OUT.mkdir(exist_ok=True, parents=True)
    before = protected_hashes()
    summary = {"configuration": CONFIG, "software": runtime_signature(), "mice": {}}
    metrics = []
    for mouse, folder in BLOCKS.items():
        print(f"Loading Mouse {mouse}: {folder.name}", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block = tdt.read_block(
                str(folder), evtype=["streams"], store=["LFP1", "eS1r"]
            )
        lfp = block.streams["LFP1"]
        values = np.atleast_2d(lfp.data).astype(np.float64)
        fs, channels = float(lfp.fs), np.asarray(lfp.channel, dtype=int)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite acquired samples")
        onsets, centers = command_centers(block)
        if not all(np.count_nonzero(channels == cid) == 1 for cid in SITES):
            raise ValueError("LFP1 must contain one CH2 (LP) and one CH3 (SC)")
        offsets = np.arange(-round(0.25 * fs), round(0.5 * fs))
        time = offsets / fs
        acquired_epochs = extract(values, centers, offsets)
        destination = OUT / f"mouse_{mouse}"
        destination.mkdir(exist_ok=True)
        arrays = {
            "time_s": time,
            "channel_ids": channels,
            "command_onset_s": onsets,
            "lfp_sample_centers": centers,
            "fs_Hz": fs,
            "source_block": str(folder),
            "configuration_json": json.dumps(CONFIG),
        }
        padding_check = {}
        for mode in MODES:
            filtered = apply_filter(values, fs, mode)
            epochs = extract(filtered, centers, offsets)
            corrected, baselines = baseline_correct(epochs, time)
            mean, sd = corrected.mean(axis=0), corrected.std(axis=0, ddof=0)
            if (
                np.max(
                    np.abs(corrected[..., (time >= -0.25) & (time < -0.05)].mean(-1))
                )
                >= 1e-7
            ):
                raise RuntimeError("Baseline correction check failed")
            arrays[f"{mode}_epochs_uV"] = epochs
            arrays[f"{mode}_baseline_corrected_uV"] = corrected
            arrays[f"{mode}_baseline_uV"] = baselines
            arrays[f"{mode}_mean_uV"] = mean
            arrays[f"{mode}_sd_uV"] = sd
            # Independently crop long context around each event; quantify boundary sensitivity.
            if mode != "no_extra_filter":
                padded = []
                pad = round(4 * fs)
                for center in centers:
                    start, stop = (
                        center + offsets[0] - pad,
                        center + offsets[-1] + pad + 1,
                    )
                    if start < 0 or stop > values.shape[-1]:
                        raise ValueError(
                            "Boundary-sensitivity check requires 4 s context around every epoch"
                        )
                    part = apply_filter(values[:, start:stop], fs, mode)
                    padded.append(part[:, center - start + offsets] * 1e6)
                padded = np.stack(padded)
                padded_bc, _ = baseline_correct(padded, time)
                padding_check[mode] = {
                    "context_s_each_side_of_epoch": 4,
                    "max_abs_unbaselined_difference_uV": float(
                        np.max(np.abs(padded - epochs))
                    ),
                    "max_abs_baselined_difference_uV": float(
                        np.max(np.abs(padded_bc - corrected))
                    ),
                }
            for channel, site in SITES.items():
                ix = int(np.flatnonzero(channels == channel)[0])
                post = (time >= 0.02) & (time <= 0.15)
                pre = (time >= -0.1) & (time < 0)
                late = (time >= 0.2) & (time <= 0.4)
                metrics.append(
                    {
                        "mouse": mouse,
                        "channel": channel,
                        "site": site,
                        "mode": mode,
                        "trials": len(centers),
                        "mean_post20_150ms_min_uV": float(mean[ix, post].min()),
                        "mean_post20_150ms_max_uV": float(mean[ix, post].max()),
                        "mean_post20_150ms_sine60_amplitude_uV": sine_amplitude(
                            time[post], mean[ix, post]
                        ),
                        "mean_prestim100ms_max_abs_uV": float(
                            np.max(np.abs(mean[ix, pre]))
                        ),
                        "mean_late200_400ms_min_uV": float(mean[ix, late].min()),
                        "mean_late200_400ms_max_uV": float(mean[ix, late].max()),
                    }
                )
        # Diagnostic only: quantify the difference from filtering each short epoch directly.
        short_filtered = apply_filter(acquired_epochs, fs, "bandpass")
        short_bc, _ = baseline_correct(short_filtered, time)
        np.savez_compressed(destination / "filtered_epochs.npz", **arrays)
        for cid, site in SITES.items():
            ix = int(np.flatnonzero(channels == cid)[0])
            export = [time]
            headers = ["time_from_command_s"]
            for mode in MODES:
                export.extend(
                    [arrays[f"{mode}_mean_uV"][ix], arrays[f"{mode}_sd_uV"][ix]]
                )
                headers.extend([f"{mode}_mean_uV", f"{mode}_sd_uV"])
            np.savetxt(
                destination / f"CH{cid}_{site}_means.csv",
                np.column_stack(export),
                delimiter=",",
                header=",".join(headers),
                comments="",
            )
        sos, (b, a) = coefficients(fs)
        frequencies = np.array([0.5, 1.0, 10.0, 30.5, 60.0, 120.0])
        _, hbp = freqz_sos(sos, worN=frequencies, fs=fs)
        _, hn = freqz(b, a, worN=frequencies, fs=fs)
        summary["mice"][str(mouse)] = {
            "block": str(folder),
            "mouse_identity": "Lab-confirmed mapping; both blocks share acquisition Subject label",
            "mapping": {"2": "LP", "3": "SC"},
            "stimulated_site": "ACC (lab-confirmed)",
            "fs_Hz": fs,
            "trials": len(centers),
            "channels": channels.tolist(),
            "padding_sensitivity": padding_check,
            "short_epoch_vs_continuous_BP_mean_max_abs_difference_uV": float(
                np.max(np.abs(short_bc.mean(0) - arrays["bandpass_mean_uV"]))
            ),
            "frequency_Hz": frequencies.tolist(),
            "bandpass_forward_backward_amplitude_gain": (np.abs(hbp) ** 2).tolist(),
            "notch_forward_backward_amplitude_gain": (np.abs(hn) ** 2).tolist(),
            "bandpass_sos": sos.tolist(),
            "notch_b": b.tolist(),
            "notch_a": a.tolist(),
            "read_warnings": [str(w.message) for w in caught],
        }
    if protected_hashes() != before:
        raise RuntimeError("Source recording changed during processing")
    summary["protected_sha256"] = before
    summary["source_unchanged"] = True
    summary["result_sha256"] = {
        str(mouse): digest(OUT / f"mouse_{mouse}/filtered_epochs.npz")
        for mouse in BLOCKS
    }
    summary["limitations"] = [
        "Filter comparison only: all stimulus transients retained, no physiological peak assignments.",
        "Forward-backward filters can spread the retained stimulus transient before and after command zero.",
        "Filter settings follow Dr. Bin Gu's example; event extraction and baseline are adapted to these recordings.",
        "The source LFP already contains acquisition filtering; no_extra_filter is not raw electrode voltage.",
        "60Hz fit on the evoked trace describes oscillation, not uniquely powerline interference.",
    ]
    write_json(OUT / "processing_summary.json", summary)
    with (OUT / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(
        f"Processed {len(BLOCKS)} recording(s), 30 trials each, four filter branches; source files unchanged.",
        flush=True,
    )


def ensure_results(rebuild=False):
    report = OUT / "processing_summary.json"
    if (
        rebuild
        or not report.exists()
        or not all((OUT / f"mouse_{m}/filtered_epochs.npz").exists() for m in BLOCKS)
    ):
        build()
    else:
        saved = json.loads(report.read_text(encoding="utf-8"))
        results = {
            str(mouse): digest(OUT / f"mouse_{mouse}/filtered_epochs.npz")
            for mouse in BLOCKS
        }
        if (
            saved["configuration"] != CONFIG
            or protected_hashes() != saved["protected_sha256"]
            or saved.get("software") != runtime_signature()
            or saved.get("result_sha256") != results
        ):
            raise ValueError(
                "Analysis inputs/configuration changed. Rebuild with filter_ccep.py --rebuild or artifact_ccep.py --build."
            )


def draw_axis(ax, time, trials, title, xlim, ylim=0.001):
    mean, sd = trials.mean(0), trials.std(0, ddof=0)
    ax.plot(time, trials.T, color=(0.85, 0.85, 0.85), linewidth=0.3)
    ax.axvline(0, color="r", linewidth=1)
    ax.plot(time, mean, color="b", linewidth=1)
    ax.plot(time, mean + sd, "b", time, mean - sd, "b", linewidth=0.1)
    ax.fill_between(time, mean + sd, mean - sd, color="#64788C", alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(-ylim, ylim)
    visible = (time >= xlim[0]) & (time <= xlim[1])
    if np.any(np.abs(trials[:, visible]) > ylim) or np.any(
        np.abs(mean[visible]) + sd[visible] > ylim
    ):
        ax.text(
            0.98,
            0.97,
            "Voltage zoom: traces exceed limits",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time from stimulation command (s)")
    ax.set_ylabel("Voltage (V)")


def plot(mice, mode="bandpass", show=True, compare=False, xlim=(-0.1, 0.4), ylim=0.001):
    import matplotlib

    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    for mouse in mice:
        directory = OUT / f"mouse_{mouse}"
        with np.load(directory / "filtered_epochs.npz", allow_pickle=False) as z:
            time, channels = z["time_s"], z["channel_ids"]
            for cid, site in SITES.items():
                ix = int(np.flatnonzero(channels == cid)[0])
                if compare:
                    modes = ("no_extra_filter", "bandpass", "notch60_bandpass")
                    fig, axs = plt.subplots(
                        1, 3, figsize=(13, 5), sharex=True, sharey=True
                    )
                    for ax, branch in zip(axs, modes):
                        draw_axis(
                            ax,
                            time,
                            z[f"{branch}_baseline_corrected_uV"][:, ix] / 1e6,
                            LABELS[branch],
                            xlim,
                            ylim,
                        )
                    fig.suptitle(
                        f"Mouse {mouse} - CH{cid} ({site}) - 30 trials", fontsize=12
                    )
                    name = f"M{mouse}_CH{cid}_{site}_comparison"
                else:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    draw_axis(
                        ax,
                        time,
                        z[f"{mode}_baseline_corrected_uV"][:, ix] / 1e6,
                        f"CH{cid} ({site}) - 30 trials - Mouse {mouse}\n{LABELS[mode]}",
                        xlim,
                        ylim,
                    )
                    name = f"M{mouse}_CH{cid}_{site}_{mode}"
                fig.canvas.manager.set_window_title(name)
                fig.text(
                    0.5,
                    0.012,
                    "Mean +/- SD; stimulus transient retained.",
                    ha="center",
                    fontsize=9,
                )
                fig.tight_layout(rect=(0, 0.035, 1, 1))
                fig.savefig(directory / f"{name}.png", dpi=140)
                if not show:
                    plt.close(fig)
    if show:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", choices=["1", "2", "all"], default="all")
    parser.add_argument("--mode", choices=MODES, default="bandpass")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--xmax", type=float, default=0.4)
    parser.add_argument(
        "--ylim",
        type=float,
        default=0.001,
        help="Symmetric voltage limit in V (default: 0.001)",
    )
    add_path_arguments(parser)
    args = parser.parse_args(argv)
    if not (0.05 <= args.xmax <= 0.49):
        parser.error("--xmax must be between .05 and .49 seconds")
    if not np.isfinite(args.ylim) or args.ylim <= 0:
        parser.error("--ylim must be positive and finite")
    mice = [1, 2] if args.mouse == "all" else [int(args.mouse)]
    configure_paths(args, mice)
    ensure_results(args.rebuild)
    if args.build_only:
        for mode in MODES:
            plot(mice, mode, show=False, xlim=(-0.1, args.xmax), ylim=args.ylim)
        plot(mice, show=False, compare=True, xlim=(-0.1, args.xmax), ylim=args.ylim)
    else:
        plot(
            mice,
            args.mode,
            compare=args.compare,
            xlim=(-0.1, args.xmax),
            ylim=args.ylim,
        )


if __name__ == "__main__":
    main()
