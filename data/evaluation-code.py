"""CPU-only evaluation of all 180 DiDi sequences for five complete baselines.

Run inside the existing VOT 0.7.1 evaluation container, with all input archives
in the current directory: python -B full_eval.py [snapshot-inputs.json].
Only full-evaluation/ and temporary scratch directories are written. Raw
prediction masks are streamed as RLE; only up to three masks per trajectory
are decoded by VOT to independently validate the tight-header box conversion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True

OLD_MODELS = ("DAM4SAM", "SAMURAI")
NEW_MODELS = ("SAM2.1", "HiM2SAM", "SENTRY-D4S")
PROTOCOLS = ("original",)
METRICS = ("Q", "A", "R")
TOLERANCE = 1e-9
EXPECTED_KEYS = {f"s{index:03d}" for index in range(180)}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    # Matches evaluate_canonical.py's filtering and NumPy mean convention.
    return float(np.asarray(values, dtype=float).mean()) if values else None


def member_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "Unsafe archive member: " + repr(name))
    return "/".join(part for part in path.parts if part != ".")


class Archive:
    """Read named regular members without extracting prediction archives."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = tarfile.open(self.path, "r:*")
        self.members = {}
        for member in self.archive.getmembers():
            name = member_name(member.name)
            require(member.isfile() or member.isdir(), "Non-regular tar member: " + member.name)
            if member.isfile():
                require(name not in self.members, "Duplicate tar member: " + name)
                self.members[name] = member

    def find(self, candidates, optional=False):
        matches = [name for name in candidates if name in self.members]
        require(len(matches) <= 1, "Ambiguous archive members: " + repr(matches))
        if not matches:
            require(optional, "Missing member in " + str(self.path) + ": " + repr(candidates))
            return None
        return matches[0]

    def text(self, name):
        with self.archive.extractfile(self.members[name]) as stream:
            return stream.read().decode("utf-8-sig")

    def lines(self, name):
        with self.archive.extractfile(self.members[name]) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8-sig") as text:
                yield from text

    def close(self):
        self.archive.close()


def extract_metadata(path, destination):
    """Copy metadata into scratch, then hide backup GT from the VOT loader."""
    seen = set()
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = member_name(item.filename)
            require(name not in seen, "Duplicate ZIP member: " + name)
            seen.add(name)
            require((item.external_attr >> 16) & 0o170000 != 0o120000,
                    "Metadata ZIP contains a symbolic link")
            target = destination / name
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    candidates = [path.parent for path in destination.rglob("list.txt")
                  if path.parent.name == "view"]
    require(len(candidates) == 1, "Expected one metadata view/list.txt")
    view = candidates[0]
    active_hashes = {path.parent.name: sha(path) for path in view.glob("*/groundtruth.txt")}
    backups = []
    hidden = destination / "backups-outside-loader"
    for path in sorted(view.glob("*/groundtruth_*.txt")):
        require(path.name == "groundtruth_old.txt", "Unexpected multiobject GT: " + str(path))
        target = hidden / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        backups.append({"sequence": path.parent.name, "backup_sha256": sha(path)})
        path.rename(target)
    require(all(sha(view / name / "groundtruth.txt") == digest
                for name, digest in active_hashes.items()), "Scratch active GT changed")
    return view, active_hashes, backups


def header_box(line):
    """Validate tight RLE without allocating pixels; reproduce VOT's max-min box."""
    line = line.strip()
    require(line.startswith("m"), "New trajectory must contain only raw mask RLE")
    tokens = line[1:].split(",")
    require(len(tokens) >= 5, "Incomplete mask RLE")
    values = [int(token) for token in tokens]
    x, y, width, height = values[:4]
    runs = values[4:]
    require(width >= 0 and height >= 0 and all(value >= 0 for value in runs),
            "Negative RLE dimensions or runs")
    require(sum(runs) == width * height, "RLE run sum does not match dimensions")
    foreground = sum(runs[1::2])
    if not foreground:
        # move_didi_results.py explicitly uses this for every empty Mask.
        return Rectangle(0, 0, 0, 0), 0
    require(width > 0 and height > 0, "Nonempty zero-sized mask")
    left, right, top, bottom, position = width, -1, height, -1, 0
    for index, length in enumerate(runs):
        if index % 2 and length:
            start_y, start_x = divmod(position, width)
            end_y, end_x = divmod(position + length - 1, width)
            top, bottom = min(top, start_y), max(bottom, end_y)
            if start_y != end_y:
                left, right = 0, width - 1
            else:
                left, right = min(left, start_x), max(right, end_x)
        position += length
    require((left, top, right, bottom) == (0, 0, width - 1, height - 1),
            "Mask RLE header is not tightly cropped")
    return Rectangle(x, y, width - 1, height - 1), foreground


def rectangle_values(region):
    return tuple(float(value) for value in (region.x, region.y, region.width, region.height))


def converted_predictions(archive, sequence, frames):
    name = archive.find([f"{sequence}/{sequence}.txt", f"result/{sequence}/{sequence}.txt"])
    sampled = sorted({0, frames // 2, frames - 1}) if frames else []
    trajectory, checks, raw_hash = [], [], hashlib.sha256()
    empty_count = 0
    for index, raw in enumerate(archive.lines(name)):
        require(index < frames, "New trajectory has excess frames: " + sequence)
        raw_hash.update(raw.encode("utf-8"))
        region, foreground = header_box(raw)
        empty_count += int(foreground == 0)
        if index in sampled:
            native = parse_region(raw.strip())
            require(native.type == RegionType.MASK, "Native parser did not produce Mask")
            expected = (Rectangle(0, 0, 0, 0) if native.is_empty()
                        else native.convert(RegionType.RECTANGLE))
            require(rectangle_values(expected) == rectangle_values(region),
                    "Native/header conversion mismatch: " + sequence + " frame " + str(index + 1))
            checks.append({"frame": index + 1, "native_matches_header": True,
                           "empty": not foreground, "rectangle": rectangle_values(region)})
        trajectory.append(Special(1) if index == 0 else region)
    require(len(trajectory) == frames, "New trajectory frame count mismatch: " + sequence)
    return trajectory, {"member": name, "raw_text_sha256": raw_hash.hexdigest(),
                        "raw_empty_frames": empty_count, "native_conversion_samples": checks,
                        "sample_count": len(checks), "sample_limit": 3,
                        "all_nonempty_headers_proved_tight": True,
                        "frame_zero_converted_to_special_1": True}


def score(prediction, groundtruth, bounds, ignore):
    # Keep these calls aligned with evaluate_canonical.py:89-100, including the
    # initialization Special(1) quirk at burnin=0 and VOT's handling of NaN GT.
    q, qi = gather_overlaps(prediction, groundtruth, burnin=0, ignore_unknown=False,
                            ignore_invisible=False, bounds=bounds, ignore_masks=ignore)
    a, ai = gather_overlaps(prediction, groundtruth, burnin=0, ignore_unknown=False,
                            ignore_invisible=True, threshold=0, bounds=bounds, ignore_masks=ignore)
    counts = count_frames(prediction, groundtruth, bounds=bounds, ignore_masks=ignore)
    tracking, failure, miss, hallucination, notice = counts
    denominator = tracking + failure + miss
    return {"Q": finite_mean(q), "Q_scored_frames": len(q),
            "A": finite_mean(a) if len(a) else 0.0, "A_scored_frames": len(a),
            "R": float(tracking / denominator) if denominator else None,
            "counts_T_F_M_H_N": [int(value) for value in counts]}


def compare_scores(actual, expected, label):
    errors = {}
    for metric in METRICS:
        require(metric in expected, "Missing reference metric: " + label + "/" + metric)
        left, right = actual[metric], expected[metric]
        if left is None or right is None:
            require(left is None and right is None, "Reference null mismatch: " + label + "/" + metric)
            errors[metric] = None
        else:
            difference = abs(float(left) - float(right))
            require(math.isfinite(difference) and difference < TOLERANCE,
                    "Reference mismatch " + label + "/" + metric + ": " + repr((left, right)))
            errors[metric] = difference
    return errors


def aggregate(rows, models, protocol):
    return {model: {metric: finite_mean(row["protocols"][protocol][model][metric] for row in rows)
                    for metric in METRICS} for model in models}


def validate_full_manifest(inputs):
    entries = inputs["new_results"]
    require(isinstance(entries, list) and len(entries) == 540, "Full evaluation requires exactly 540 new trajectories")
    require(Counter(entry["model"] for entry in entries) == Counter({model: 180 for model in NEW_MODELS}),
            "Full evaluation requires exactly 180 trajectories for each new model")
    require(len({entry["archive"] for entry in entries}) == 540, "Duplicate archive basenames in full snapshot")
    mappings = {}
    for model in NEW_MODELS:
        selected = [entry for entry in entries if entry["model"] == model]
        require({entry["key"] for entry in selected} == EXPECTED_KEYS, "Missing or duplicate keys for " + model)
        require(len({entry["sequence"] for entry in selected}) == 180, "Duplicate sequences for " + model)
        mappings[model] = {entry["key"]: entry["sequence"] for entry in selected}
        for entry in selected:
            require(isinstance(entry["frames"], int) and not isinstance(entry["frames"], bool) and entry["frames"] > 0,
                    "Invalid snapshot frame count")
            require(re.fullmatch(r"[0-9a-f]{64}", entry["expected_trajectory_sha256"]) is not None,
                    "Invalid or missing frozen trajectory SHA256")
            require(entry["sequence"] and PurePosixPath(entry["sequence"]).name == entry["sequence"]
                    and "\\" not in entry["sequence"], "Invalid sequence basename")
            pattern = re.escape(entry["key"] + "-" + model) + r"\.\d+\.\d+\.tar\.gz"
            require(re.fullmatch(pattern, entry["archive"]) is not None, "Invalid prediction archive basename")
    require(all(mapping == mappings[NEW_MODELS[0]] for mapping in mappings.values()),
            "New models disagree about key-to-sequence mapping")


def validate_reference(reference, prior):
    require(set(PROTOCOLS).issubset(reference), "Reference must cover both evaluation protocols")
    for protocol in PROTOCOLS:
        require(set(reference[protocol]) == set(prior), "Reference sequence coverage mismatch: " + protocol)
        for sequence_name in prior:
            for model in OLD_MODELS:
                require(model in reference[protocol][sequence_name], "Missing old reference model")
                for metric in METRICS:
                    require(metric in reference[protocol][sequence_name][model], "Missing reference Q/A/R metric")
                    value = reference[protocol][sequence_name][model][metric]
                    require(value is None or (isinstance(value, (int, float)) and math.isfinite(value)),
                            "Invalid reference score")


def make_summary(rows):
    require(len(rows) == 180 and {row["key"] for row in rows} == EXPECTED_KEYS,
            "Full summary requires all 180 unique sequences")
    require(all(set(row["available_new_models"]) == set(NEW_MODELS) for row in rows),
            "Full summary requires all three new models on every sequence")
    summary = {"scope": "Complete fixed evaluation: all five models on the same 180 DiDi sequences",
               "complete_coverage": True, "all_five_models_comparable": True,
               "common_sequence_count": 180,
               "aggregation": "Unweighted sequence mean; finite values only, as canonical evaluation",
               "protocols": {}}
    for protocol in PROTOCOLS:
        comparisons = {}
        for model in NEW_MODELS:
            paired = {}
            for baseline in OLD_MODELS:
                paired[baseline] = {}
                for metric in METRICS:
                    pairs = [(row["protocols"][protocol][model][metric],
                              row["protocols"][protocol][baseline][metric]) for row in rows]
                    pairs = [(new, old) for new, old in pairs if new is not None and old is not None
                             and math.isfinite(float(new)) and math.isfinite(float(old))]
                    paired[baseline][metric] = {
                        "paired_sequence_count": len(pairs),
                        "new_mean": finite_mean(pair[0] for pair in pairs),
                        "baseline_mean": finite_mean(pair[1] for pair in pairs),
                        "new_minus_baseline": finite_mean(new - old for new, old in pairs)}
            comparisons[model] = {"sequence_count": 180, "sequence_keys": [row["key"] for row in rows],
                                  "means": aggregate(rows, (model,) + OLD_MODELS, protocol),
                                  "paired_comparisons": paired}
        counts = {model: {metric: sum(row["protocols"][protocol][model][metric] is not None
                                     and math.isfinite(float(row["protocols"][protocol][model][metric])) for row in rows)
                          for metric in METRICS} for model in NEW_MODELS + OLD_MODELS}
        summary["protocols"][protocol] = {
            "complete_model_comparisons": comparisons,
            "five_model_comparison": {
                "sequence_count": 180, "sequence_keys": [row["key"] for row in rows],
                "means": aggregate(rows, NEW_MODELS + OLD_MODELS, protocol),
                "finite_sequence_counts": counts,
                "interpretation": "All five model means use the same complete 180-sequence set"}}
    return summary

def run(args, output, status):
    global np, Rectangle, Special, RegionType, parse_region, gather_overlaps, count_frames
    import numpy as np
    from vot.dataset import load_dataset
    from vot.region import Rectangle, Special, RegionType
    from vot.region.io import parse_region, read_trajectory
    from vot.analysis.accuracy import gather_overlaps
    from vot.analysis.longterm import count_frames

    version = importlib.metadata.version("vot-toolkit")
    require(version == "0.7.1", "Expected existing VOT 0.7.1 container, found " + version)
    manifest_path = args.inputs.resolve()
    inputs = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    old_path = base / inputs.get("old_archive", "evaluation-results-canonical.7225695.tar.gz")
    metadata_path = base / inputs.get("metadata_archive", "evaluation-metadata.zip")
    reference_path = args.reference.resolve() if args.reference else base / "reference-scores.json"
    require(not args.reference or reference_path.is_file(), "Explicit reference file is missing: " + str(reference_path))
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    require(reference_path.is_file(), "Full evaluation requires reference-scores.json")
    require(reference is not None, "Full evaluation requires both old reference protocols")
    validate_full_manifest(inputs)
    entries = inputs["new_results"]
    indexed = {}
    for entry in entries:
        require(entry["model"] in NEW_MODELS, "Unexpected new model: " + entry["model"])
        require(Path(entry["archive"]).name == entry["archive"], "New archive must be a transferred basename")
        require((entry["model"], entry["sequence"]) not in indexed, "Duplicate model/sequence in snapshot")
        indexed[entry["model"], entry["sequence"]] = entry
    status.update(phase="prepare-scratch-metadata", vot_toolkit=version,
                  new_trajectory_count=len(entries), reference_scores_present=reference is not None,
                  manifest_sha256=sha(manifest_path), old_archive_sha256=sha(old_path),
                  metadata_archive_sha256=sha(metadata_path))
    if reference is not None:
        status["reference_scores_sha256"] = sha(reference_path)
    for name, path in (("old_archive", old_path), ("metadata_archive", metadata_path),
                       ("reference_scores", reference_path)):
        require(sha(path) == inputs["expected_input_sha256"][name], "Frozen prior input SHA256 mismatch: " + name)
    write_json(output / "status.json", status)
    rows = []
    with tempfile.TemporaryDirectory(prefix="didi-full-eval-") as scratch:
        view, active_hashes, backups = extract_metadata(metadata_path, Path(scratch) / "dataset")
        dataset = load_dataset(str(view))
        sequences = {sequence.name: sequence for sequence in dataset}
        old_archive = Archive(old_path)
        try:
            prior_member = old_archive.find(["evaluation-results/per-sequence.json", "per-sequence.json"])
            prior_rows = json.loads(old_archive.text(prior_member))
            prior = {row["sequence"]: row for row in prior_rows}
            require(len(prior_rows) == len(prior) == 180, "Canonical prior must contain exactly 180 unique sequences")
            require({row["key"] for row in prior_rows} == EXPECTED_KEYS, "Canonical prior key coverage mismatch")
            require(set(sequences) == set(prior), "Metadata/canonical sequence coverage mismatch")
            validate_reference(reference, prior)
            for entry in entries:
                old = prior[entry["sequence"]]
                require(entry["key"] == old["key"] and entry["frames"] == old["frames"],
                        "Full snapshot frame/key mapping differs from canonical prior")
            selected = sorted({entry["sequence"] for entry in entries}, key=lambda name: prior[name]["key"])
            for sequence_name in selected:
                status.update(phase="score-sequence", current_sequence=sequence_name, completed_sequences=len(rows))
                write_json(output / "status.json", status)
                sequence = sequences[sequence_name]
                old = prior[sequence_name]
                require(set(sequence.objects()) == {"object"}, "Unexpected VOT object IDs: " + sequence_name)
                require(len(sequence) == old["frames"], "Metadata length mismatch: " + sequence_name)
                require(tuple(sequence.size) == (old["width"], old["height"]), "Metadata dimensions mismatch")
                gt = sequence.groundtruth()
                direct = read_trajectory(str(view / sequence_name / "groundtruth.txt"))
                require(len(gt) == len(direct) == old["frames"], "Active GT length mismatch")
                require(all(left.type == right.type for left, right in zip(gt, direct)), "VOT GT type mismatch")
                available = [model for model in NEW_MODELS if (model, sequence_name) in indexed]
                row = {name: old[name] for name in ("key", "sequence", "frames", "width", "height")}
                row.update(available_new_models=available, active_gt_sha256=active_hashes[sequence_name],
                           gt_types=dict(Counter(region.type.name for region in gt)),
                           protocols={protocol: {} for protocol in PROTOCOLS},
                           old_score_checks={}, new_predictions={})
                ignore = sequence.object("_ignore")
                for model in OLD_MODELS + tuple(available):
                    if model in OLD_MODELS:
                        suffix = f"workspace/results/{model}/baseline/{sequence_name}/{sequence_name}_001.txt"
                        member = old_archive.find(["evaluation-results/" + suffix, suffix])
                        prediction = [parse_region(line.strip()) for line in old_archive.lines(member)]
                        require(len(prediction) == len(gt), "Old trajectory length mismatch")
                        require(isinstance(prediction[0], Special) and prediction[0].code == 1,
                                "Old frame zero is not Special(1)")
                    else:
                        entry = indexed[model, sequence_name]
                        require(entry["key"] == old["key"], "Snapshot key/sequence mismatch")
                        archive_path = base / entry["archive"]
                        archive = Archive(archive_path)
                        try:
                            prediction, conversion = converted_predictions(archive, sequence_name, len(gt))
                        finally:
                            archive.close()
                        if "expected_trajectory_sha256" in entry:
                            require(conversion["raw_text_sha256"] == entry["expected_trajectory_sha256"],
                                    "Snapshot trajectory SHA256 mismatch: " + model + "/" + sequence_name)
                            conversion["snapshot_trajectory_sha256_matches"] = True
                        row["new_predictions"][model] = dict(conversion, archive=entry["archive"],
                                                            archive_sha256=sha(archive_path))
                    for protocol, truth in (("original", gt),):
                        measured = score(prediction, truth, sequence.size, ignore)
                        row["protocols"][protocol][model] = measured
                        if model in OLD_MODELS:
                            checks = row["old_score_checks"].setdefault(model, {})
                            if protocol == "original":
                                checks["canonical_per_sequence"] = compare_scores(
                                    measured, old[model], sequence_name + "/" + model + "/canonical")
                            if reference is not None:
                                checks[protocol + "_reference"] = compare_scores(
                                    measured, reference[protocol][sequence_name][model],
                                    sequence_name + "/" + model + "/" + protocol)
                rows.append(row)
                write_json(output / "per-sequence.json", rows)
                print("SCORED", len(rows), "/", len(selected), old["key"], sequence_name,
                      ",".join(available), flush=True)
            require(all(sha(view / name / "groundtruth.txt") == digest
                        for name, digest in active_hashes.items()), "Active GT changed during evaluation")
        finally:
            old_archive.close()
        summary = make_summary(rows)
        require(len(rows) == 180, "Full evaluation sequence count mismatch")
        require(sum(len(row["new_predictions"]) for row in rows) == 540, "Full evaluation trajectory count mismatch")
        summary.update(snapshot_input_sha256=status["manifest_sha256"],
                       old_trajectory_count=360, protocol_trajectory_score_count=900,
                       new_trajectory_count=len(entries), union_sequence_count=len(rows),
                       active_gt_unchanged=True, backup_gt_files_hidden_in_scratch=len(backups),
                       original_old_scores_match_canonical=True,
                       old_reference_protocols_checked=list(PROTOCOLS) if reference is not None else [],
                       score_tolerance_strictly_less_than=TOLERANCE,
                       no_model_inference=True, no_gpu=True,
                       initialization="All predictions use Special(1) at frame zero; canonical burnin=0 retained",
                       original="Converted prediction boxes versus unchanged active GT regions",
                       report_protocol="original")
        write_json(output / "per-sequence.json", rows)
        write_json(output / "summary.json", summary)
    status.update(status="verified", phase="complete", completed_sequences=len(rows),
                  scored_new_trajectories=len(entries), scored_old_trajectories=2 * len(rows),
                  native_conversion_sample_count=sum(item["sample_count"] for row in rows
                                                     for item in row["new_predictions"].values()),
                  active_gt_unchanged=True, original_old_scores_match_canonical=True,
                  reference_protocols_checked=list(PROTOCOLS) if reference is not None else [],
                  finished_utc=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="?", type=Path, default=Path("snapshot-inputs.json"))
    parser.add_argument("--reference", type=Path, help="Default: reference-scores.json beside input manifest")
    parser.add_argument("--output", type=Path, default=Path("full-evaluation"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "phase": "start", "started_utc": now(),
              "no_model_inference": True, "cuda_visible_devices": "", "script_sha256": sha(__file__)}
    started = time.monotonic()
    write_json(output / "status.json", status)
    try:
        # Avoid leaving prior successful JSON beside a failed rerun's status.
        require(not (output / "summary.json").exists() and not (output / "per-sequence.json").exists(),
                "Output already contains evaluation data; select a fresh --output directory")
        with tempfile.TemporaryDirectory(prefix="didi-eval-caches-") as cache:
            os.environ["NUMBA_CACHE_DIR"] = str(Path(cache) / "numba")
            os.environ["MPLCONFIGDIR"] = str(Path(cache) / "matplotlib")
            run(args, output, status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc(), finished_utc=now())
        traceback.print_exc()
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)
    return 0 if status["status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
