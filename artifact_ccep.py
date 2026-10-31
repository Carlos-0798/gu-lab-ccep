"""Compare bandpass traces before and after Dr. Bin Gu's 2 ms copy method.

Replacement is aligned to the saved command. It does not remove the later
transient in these recordings. The 1 mV rejection rule is diagnostic only.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import tdt

import filter_ccep as base

OUT = base.ROOT / "results/artifact"
CONFIG = {
    "copy_window_s": [0, 0.002],
    "copy_source_s": [-0.002, 0],
    "time_zero": "Saved eS1r command; physical pulse arrival unverified",
    "main_comparison": "Copy in continuous acquired signal BEFORE full-record bandpass",
    "bandpass": "butter(1, [0.5, 30.5], fs=fs, output=sos); sosfiltfilt axis=-1",
    "epoch_s": [-0.25, 0.5],
    "baseline_s": [-0.25, -0.05],
    "sd_ddof": 0,
    "trial_rejection_applied": False,
    "trial_rejection_diagnostic_V": 0.001,
    "trial_rejection_basis": "Acquired samples before offline filtering or baseline subtraction; -250 to +250 ms",
    "early_peak_basis": "Absolute maximum in baseline-corrected acquired signal; 0 to 20 ms",
    "channels_replaced": [2, 3],
    "notch_added": False,
    "reference_check": "Separate matched short-epoch pair (-250 to +250 ms), copy then bandpass",
    "source_adaptations": [
        "LFP1 recordings and eS1r command centers replace the example's stream/event names",
        "Copy window starts at the command sample in the asymmetric epoch",
        "Optional rejection not applied to comparison; its all-rejected result is reported",
        "Main pair keeps existing full-record filtering and per-trial baseline unchanged",
        "Short-epoch diagnostic uses shared mean-baseline subtraction like example",
        "SD is computed across corrected trials; physiological peak labels are not assigned",
    ],
}


def copy_pre_event(values, centers, rows, count):
    if count <= 0:
        raise ValueError("Copy window must contain at least one sample")
    result = np.array(values, dtype=np.float64, copy=True)
    for center in centers:
        if center < count or center + count > result.shape[-1]:
            raise ValueError("Insufficient event context")
        for row in rows:
            result[row, center : center + count] = values[row, center - count : center]
    return result


def cache_path(mouse):
    return OUT / f"mouse_{mouse}/artifact_comparison.npz"


def protected():
    hashes = base.protected_hashes()
    for mouse in base.BLOCKS:
        path = base.OUT / f"mouse_{mouse}/filtered_epochs.npz"
        hashes[f"filter/mouse_{mouse}/filtered_epochs.npz"] = base.digest(path)
    return hashes


def processing_signature():
    return {
        "base": base.runtime_signature(),
        "base_configuration": base.CONFIG,
        "artifact_sha256": base.digest(Path(__file__)),
    }


def ensure_results(rebuild=False):
    base.ensure_results(rebuild)
    report = OUT / "processing_summary.json"
    if (
        rebuild
        or not report.exists()
        or not all(cache_path(mouse).exists() for mouse in base.BLOCKS)
    ):
        build()
        return
    saved = json.loads(report.read_text(encoding="utf-8"))
    results = {str(mouse): base.digest(cache_path(mouse)) for mouse in base.BLOCKS}
    if (
        saved["configuration"] != CONFIG
        or saved.get("software") != processing_signature()
        or protected() != saved["protected_sha256_after"]
        or saved.get("result_sha256") != results
    ):
        raise ValueError(
            "Artifact inputs/configuration changed. Use --build to regenerate results."
        )


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected()
    summary = {
        "configuration": CONFIG,
        "software": processing_signature(),
        "mice": {},
        "metrics": [],
    }
    for mouse, folder in base.BLOCKS.items():
        print(f"Mouse {mouse}: {folder.name}", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block = tdt.read_block(
                str(folder), evtype=["streams"], store=["LFP1", "eS1r"]
            )
        lfp = block.streams["LFP1"]
        values = np.atleast_2d(lfp.data).astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite acquired samples")
        fs, channels = float(lfp.fs), np.asarray(lfp.channel, dtype=int)
        onsets, centers = base.command_centers(block)
        rows = [int(np.flatnonzero(channels == cid)[0]) for cid in base.SITES]
        count = round(0.002 * fs)
        offsets = np.arange(-round(0.25 * fs), round(0.5 * fs))
        time = offsets / fs
        copied = copy_pre_event(values, centers, rows, count)
        changed_positions = np.zeros_like(values, dtype=bool)
        for center in centers:
            for row in rows:
                changed_positions[row, center : center + count] = True
                if not np.array_equal(
                    copied[row, center : center + count],
                    values[row, center - count : center],
                ):
                    raise RuntimeError("Copy-window validation failed")
        if not np.array_equal(values[~changed_positions], copied[~changed_positions]):
            raise RuntimeError("Samples outside the copy window changed")
        acquired = base.extract(values, centers, offsets)
        replaced = base.extract(copied, centers, offsets)
        raw_bc, _ = base.baseline_correct(acquired, time)
        replaced_bc, _ = base.baseline_correct(replaced, time)
        arrays = {
            "time_s": time,
            "channel_ids": channels,
            "fs_Hz": fs,
            "command_onset_s": onsets,
            "lfp_sample_centers": centers,
            "raw_epochs_uV": acquired,
            "copied_epochs_uV": replaced,
            "raw_baseline_corrected_uV": raw_bc,
            "copied_raw_baseline_corrected_uV": replaced_bc,
            "copy_sample_count": count,
            "configuration_json": json.dumps(CONFIG),
        }
        for name, signal in [("before", values), ("after", copied)]:
            epochs = base.extract(
                base.apply_filter(signal, fs, "bandpass"), centers, offsets
            )
            bc, baselines = base.baseline_correct(epochs, time)
            arrays[f"{name}_epochs_uV"] = epochs
            arrays[f"{name}_baseline_corrected_uV"] = bc
            arrays[f"{name}_baseline_uV"] = baselines
            arrays[f"{name}_mean_uV"] = bc.mean(axis=0)
            arrays[f"{name}_sd_uV"] = bc.std(axis=0, ddof=0)
        with np.load(
            base.OUT / f"mouse_{mouse}/filtered_epochs.npz", allow_pickle=False
        ) as control:
            for key in [
                "time_s",
                "channel_ids",
                "command_onset_s",
                "lfp_sample_centers",
            ]:
                if not np.array_equal(arrays[key], control[key]):
                    raise RuntimeError(f"Filter control metadata differs: {key}")
            if not np.array_equal(
                arrays["before_baseline_corrected_uV"],
                control["bandpass_baseline_corrected_uV"],
            ):
                raise RuntimeError("Filter control traces differ")
        # Matched short-epoch diagnostic: use the example's symmetric duration.
        short_offsets = np.arange(-round(0.25 * fs), round(0.25 * fs))
        short_time = short_offsets / fs
        arrays["reference_time_s"] = short_time
        for name, signal in [("before", values), ("after", copied)]:
            short = base.extract(signal, centers, short_offsets)
            result = base.apply_filter(short, fs, "bandpass")
            global_baseline = result.mean(axis=0)[:, : int(np.ceil(0.2 * fs))].mean(
                axis=-1
            )
            result = result - global_baseline[None, :, None]
            arrays[f"reference_{name}_baseline_corrected_uV"] = result
            arrays[f"reference_{name}_mean_uV"] = result.mean(axis=0)
            arrays[f"reference_{name}_sd_uV"] = result.std(axis=0, ddof=0)
        early = np.flatnonzero((time >= 0) & (time < 0.02))
        visible = (time >= -0.1) & (time <= 0.4)
        short_mask = (time >= -0.25) & (time < 0.25)
        for cid, site in base.SITES.items():
            row = int(np.flatnonzero(channels == cid)[0])
            peak_indices = early[np.argmax(np.abs(raw_bc[:, row, early]), axis=-1)]
            peak_times = time[peak_indices]
            before_peak = acquired[np.arange(30), row, peak_indices]
            after_peak = replaced[np.arange(30), row, peak_indices]
            rejected = np.any(
                np.abs(acquired[:, row, :][:, short_mask]) > 1000, axis=-1
            )
            delta = arrays["after_mean_uV"][row] - arrays["before_mean_uV"][row]
            ref_delta = (
                arrays["reference_after_mean_uV"][row]
                - arrays["reference_before_mean_uV"][row]
            )
            metric = {
                "mouse": mouse,
                "channel": cid,
                "site": site,
                "trials_compared": 30,
                "strict_1mV_rejected": int(rejected.sum()),
                "strict_1mV_retained": int((~rejected).sum()),
                "early_abspeak_time_min_ms": float(peak_times.min() * 1000),
                "early_abspeak_time_max_ms": float(peak_times.max() * 1000),
                "early_abspeak_samples_unchanged": int(
                    (before_peak == after_peak).sum()
                ),
                "mean_max_abs_change_visible_uV": float(np.max(np.abs(delta[visible]))),
                "reference_mean_max_abs_change_uV": float(np.max(np.abs(ref_delta))),
                "max_abs_trial_visible_before_uV": float(
                    np.max(
                        np.abs(
                            arrays["before_baseline_corrected_uV"][:, row, :][
                                :, visible
                            ]
                        )
                    )
                ),
            }
            summary["metrics"].append(metric)
        destination = cache_path(mouse).parent
        destination.mkdir(exist_ok=True)
        np.savez_compressed(cache_path(mouse), **arrays)
        for cid, site in base.SITES.items():
            row = int(np.flatnonzero(channels == cid)[0])
            columns = [
                time,
                arrays["before_mean_uV"][row],
                arrays["after_mean_uV"][row],
                arrays["before_sd_uV"][row],
                arrays["after_sd_uV"][row],
            ]
            np.savetxt(
                destination / f"CH{cid}_{site}_means.csv",
                np.column_stack(columns),
                delimiter=",",
                header="time_from_command_s,before_mean_uV,after_copy_mean_uV,before_sd_uV,after_copy_sd_uV",
                comments="",
            )
        summary["mice"][str(mouse)] = {
            "block": str(folder),
            "fs_Hz": fs,
            "samples_copied": count,
            "copy_duration_ms": count / fs * 1000,
            "trials": 30,
            "raw_outside_copy_unchanged": True,
            "control_matches_current_bandpass": True,
            "reader_warnings": sorted({str(w.message) for w in caught}),
        }
    after = protected()
    if before != after:
        raise RuntimeError(
            "Source recordings or filter inputs changed during processing"
        )
    summary["protected_sha256_before"] = before
    summary["protected_sha256_after"] = after
    summary["result_sha256"] = {
        str(mouse): base.digest(cache_path(mouse)) for mouse in base.BLOCKS
    }
    base.write_json(OUT / "processing_summary.json", summary)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary["metrics"][0].keys())
        writer.writeheader()
        writer.writerows(summary["metrics"])
    print(json.dumps(summary["metrics"], indent=2), flush=True)


def traces(ax, time, data, title):
    mean, sd = data.mean(axis=0), data.std(axis=0, ddof=0)
    ax.plot(time, data.T, color=(0.85, 0.85, 0.85), linewidth=0.3)
    ax.fill_between(time, mean - sd, mean + sd, color="#64788C", alpha=0.5)
    ax.plot(time, mean, color="b", linewidth=1)
    ax.plot(time, mean + sd, "b", time, mean - sd, "b", linewidth=0.1)
    ax.axvline(0, color="r", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time from command (s)")
    ax.set_ylabel("Amplitude (V)")


def plot(mice, show, view, xmax=0.4):
    import matplotlib

    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    if view in ("comparison", "all"):
        for mouse in mice:
            with np.load(cache_path(mouse), allow_pickle=False) as z:
                time = z["time_s"]
                for cid, site in base.SITES.items():
                    row = int(np.flatnonzero(z["channel_ids"] == cid)[0])
                    fig, axes = plt.subplots(
                        1, 2, figsize=(11, 5), sharex=True, sharey=True
                    )
                    for ax, name, title in zip(
                        axes,
                        ["before", "after"],
                        ["Bandpass only", "2 ms replacement + bandpass"],
                    ):
                        traces(
                            ax,
                            time,
                            z[f"{name}_baseline_corrected_uV"][:, row] / 1e6,
                            title,
                        )
                        ax.set_xlim(-0.1, xmax)
                    # Common y scale for before/after, containing all visible trials.
                    mask = (time >= -0.1) & (time <= xmax)
                    largest = (
                        max(
                            np.max(
                                np.abs(
                                    z[f"{key}_baseline_corrected_uV"][:, row, :][
                                        :, mask
                                    ]
                                )
                            )
                            for key in ["before", "after"]
                        )
                        / 1e6
                    )
                    limit = max(0.001, np.ceil(largest * 1.05 / 0.0005) * 0.0005)
                    axes[0].set_ylim(-limit, limit)
                    name = f"M{mouse}_CH{cid}_{site}_artifact_comparison"
                    fig.canvas.manager.set_window_title(name)
                    fig.suptitle(
                        f"Mouse {mouse} - CH{cid} ({site}) - 30 trials", fontsize=12
                    )
                    fig.text(
                        0.5,
                        0.012,
                        "Mean +/- SD; all trials retained. Replacement tested at saved command time.",
                        ha="center",
                        fontsize=9,
                    )
                    fig.tight_layout(rect=(0, 0.035, 1, 1))
                    fig.savefig(figures / f"{name}.png", dpi=160)
                    if not show:
                        plt.close(fig)
    if view in ("zoom", "all"):
        fig, axes = plt.subplots(
            len(mice),
            2,
            figsize=(11, 3.6 * len(mice)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for r, mouse in enumerate(mice):
            with np.load(cache_path(mouse), allow_pickle=False) as z:
                mask = (z["time_s"] >= -0.005) & (z["time_s"] <= 0.015)
                times = z["time_s"][mask] * 1000
                for col, (cid, site) in enumerate(base.SITES.items()):
                    row = int(np.flatnonzero(z["channel_ids"] == cid)[0])
                    ax = axes[r, col]
                    raw = z["raw_baseline_corrected_uV"][:, row, :][:, mask] / 1e6
                    changed = (
                        z["copied_raw_baseline_corrected_uV"][:, row, :][:, mask] / 1e6
                    )
                    ax.plot(times, raw.T, color=(0.85, 0.85, 0.85), linewidth=0.3)
                    ax.axvspan(
                        0,
                        float(z["copy_sample_count"]) / float(z["fs_Hz"]) * 1000,
                        color="#ecd99a",
                        alpha=0.45,
                        label="2 ms copy window",
                    )
                    ax.plot(
                        times,
                        raw.mean(axis=0),
                        color="k",
                        linewidth=1.4,
                        label="Before copy (mean)",
                    )
                    ax.plot(
                        times,
                        changed.mean(axis=0),
                        color="b",
                        linestyle="--",
                        linewidth=1,
                        label="After copy (mean)",
                    )
                    ax.axvline(0, color="r", linewidth=1)
                    ax.set_title(f"Mouse {mouse} - CH{cid} ({site})", fontsize=11)
                    ax.set_xlabel("Time from command (ms)")
                    ax.set_ylabel("Amplitude (V)")
                    ax.set_xlim(-5, 15)
                    largest = max(np.max(np.abs(raw)), np.max(np.abs(changed)))
                    old_limit = max(abs(x) for x in ax.get_ylim())
                    limit = max(0.04, largest * 1.05, old_limit if r or col else 0)
                    ax.set_ylim(-limit, limit)
        axes[0, 0].legend(fontsize=8, loc="upper left")
        fig.suptitle(
            "Early transient - before software bandpass - 30 trials per channel",
            fontsize=12,
        )
        name = "Early_transient_2ms_window"
        fig.canvas.manager.set_window_title(name)
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160)
        if not show:
            plt.close(fig)
    if show:
        plt.show()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--mouse", choices=["1", "2", "all"], default="all")
    parser.add_argument("--view", choices=["comparison", "zoom", "all"], default="all")
    parser.add_argument("--xmax", type=float, default=0.4)
    base.add_path_arguments(parser)
    args = parser.parse_args()
    if not (0.05 <= args.xmax <= 0.49):
        parser.error("--xmax must be between .05 and .49 seconds")
    mice = [1, 2] if args.mouse == "all" else [int(args.mouse)]
    base.configure_paths(args, mice)
    OUT = args.output_dir.resolve() / "artifact"
    ensure_results(args.build)
    plot(mice, not args.no_show, args.view, args.xmax)


if __name__ == "__main__":
    main()
