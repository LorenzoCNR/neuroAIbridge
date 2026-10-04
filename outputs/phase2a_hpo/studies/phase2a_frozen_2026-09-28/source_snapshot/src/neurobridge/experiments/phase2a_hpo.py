"""Frozen Phase-2A planning, qualification and selection boundary.

The only CLI operation is a fit-free dry run.  Training hooks from the V2
pipeline are intentionally not called: they materialize test rows/indices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_RELATIVE = Path("outputs/audits/pre_tuning_2026-09-27/PHASE2A_HPO_PROTOCOL_PROPOSAL.md")
SEALED_CANDIDATE_SHA256 = "196d0b24251ed43f2f781af635a3cb1c37abf43e5e6e7d8ed66cd261aa616f6c"
STUDY_ID = "phase2a_frozen_2026-09-28"
OBJECTIVES = ("soft", "infonce", "time_contrastive_blocks", "behavior_contrastive_blocks")
ARCHITECTURES = ("cnn1d", "transformer")
DOMAINS = {"synthetic": ("A", "B"), "real": ("TOTAL65", "A", "B")}
INITIAL_CANDIDATES = (0, 1, 2, 3)
EXTENSION_CANDIDATES = (4, 5)
SEARCH_SEED = 1101
FINALIST_SEEDS = (1201, 1301)
DATA_SEED = 42
SPLIT_SEED = 42
CHANNEL_PARTITION_SEED = 42
PAIR_SAMPLE_SEED = 100
PAIR_SAMPLE_MAX = 512
OUTPUT_RELATIVE = Path("outputs/phase2a_hpo")
CHANNEL_PARTITION_RELATIVE = Path("outputs/runs/channel_split_seed_42.json")

BOUNDS = {
    "lr": (1e-4, 3e-3),
    "weight_decay": (1e-5, 1e-3),
    "embedding_temperature": (0.05, 0.20),
    "triplet_temperature": (0.1, 2.0),
}
SCHEDULES = {
    "synthetic": {"max_updates": 4000, "validation_interval": 400, "min_updates": 800,
                  "patience": 3, "relative_min_delta": 0.001},
    "real": {"max_updates": 4000, "validation_interval": 150, "min_updates": 750,
             "patience": 5, "relative_min_delta": 0.001},
}
_CHANNELS_A = (29, 42, 18, 24, 7, 17, 27, 54, 64, 44, 51, 5, 62, 25, 32, 21,
               61, 20, 57, 48, 56, 59, 50, 4, 26, 15, 60, 23, 28, 40, 9, 37)
_CHANNELS_B = (31, 39, 16, 3, 34, 30, 10, 52, 58, 6, 38, 11, 55, 41, 46, 22, 19,
               35, 0, 47, 49, 45, 12, 53, 43, 14, 63, 2, 36, 33, 1, 13, 8)


class ProtocolViolation(RuntimeError):
    """A frozen protocol, firewall or artifact invariant failed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    objective: str
    index: int
    lr: str
    weight_decay: str
    temperature: str

    @property
    def candidate_id(self) -> str:
        return f"{self.objective}-c{self.index}"

    def numeric(self) -> dict[str, float]:
        return {"learning_rate": float(self.lr), "weight_decay": float(self.weight_decay),
                "active_temperature": float(self.temperature)}


def _candidate_zero(objective: str) -> Candidate:
    temperature = "0.1" if objective in {"soft", "infonce"} else "1.0"
    return Candidate(objective, 0, "0.001", "0.0001", temperature)


def load_sealed_candidates(protocol_path: Path) -> tuple[dict[str, tuple[Candidate, ...]], str]:
    """Read only the fenced table, verify its canonical 20-data-line seal."""
    source = protocol_path.read_text(encoding="utf-8")
    blocks = source.split("```csv\n")
    if len(blocks) != 2:
        raise ProtocolViolation("expected exactly one fenced candidate CSV")
    body = blocks[1].split("\n```", 1)
    if len(body) != 2:
        raise ProtocolViolation("candidate CSV fence is not closed")
    rows = body[0].splitlines()
    if not rows or rows[0] != "objective,candidate,lr,weight_decay,temperature" or len(rows) != 21:
        raise ProtocolViolation("candidate table header/count differs from frozen protocol")
    digest = _sha256_bytes(("\n".join(rows[1:]) + "\n").encode("utf-8"))
    if digest != SEALED_CANDIDATE_SHA256:
        raise ProtocolViolation(f"candidate checksum mismatch: {digest}")
    parsed = list(csv.DictReader(io.StringIO("\n".join(rows))))
    if len(parsed) != 20:
        raise ProtocolViolation("candidate parser did not yield 20 rows")
    result: dict[str, list[Candidate]] = {name: [_candidate_zero(name)] for name in OBJECTIVES}
    for row_number, row in enumerate(parsed):
        objective = OBJECTIVES[row_number // 5]
        index = row_number % 5 + 1
        if row["objective"] != objective or row["candidate"] != str(index):
            raise ProtocolViolation("candidate order/id differs from sealed table")
        candidate = Candidate(objective, index, row["lr"], row["weight_decay"], row["temperature"])
        temp_bounds = BOUNDS["embedding_temperature" if objective in {"soft", "infonce"} else "triplet_temperature"]
        for value, bounds in ((candidate.lr, BOUNDS["lr"]),
                              (candidate.weight_decay, BOUNDS["weight_decay"]),
                              (candidate.temperature, temp_bounds)):
            numeric = float(value)
            if not math.isfinite(numeric) or not bounds[0] <= numeric <= bounds[1]:
                raise ProtocolViolation("sealed candidate is outside approved bounds")
        result[objective].append(candidate)
    return {key: tuple(value) for key, value in result.items()}, digest


def channel_partition(project_root: Path) -> tuple[dict[str, tuple[int, ...]], str]:
    """Read the channel-only partition artifact; it has no split/test records."""
    path = project_root / CHANNEL_PARTITION_RELATIVE
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != {"A_channel_indices", "B_channel_indices", "channel_split_seed", "method",
                       "n_source_channels", "verified_complete_cover", "verified_disjoint"}:
        raise ProtocolViolation("channel partition artifact schema changed")
    a, b = tuple(values["A_channel_indices"]), tuple(values["B_channel_indices"])
    if (values["channel_split_seed"] != CHANNEL_PARTITION_SEED or
            a != _CHANNELS_A or b != _CHANNELS_B or len(a) != 32 or len(b) != 33 or
            set(a) & set(b) or set(a) | set(b) != set(range(65))):
        raise ProtocolViolation("canonical A/B partition changed")
    return {"TOTAL65": tuple(range(65)), "A": a, "B": b}, file_sha256(path)


def guarded_output_root(project_root: Path, output_root: Path | None = None) -> Path:
    """Reject V2 roots, traversal and symlink escapes before any HPO write."""
    project = project_root.resolve()
    canonical = (project / OUTPUT_RELATIVE).resolve()
    target = (output_root or canonical).resolve()
    if target != canonical:
        raise ProtocolViolation("HPO writes are restricted to outputs/phase2a_hpo")
    v2 = (project / "outputs/runs").resolve()
    if target == v2 or v2 in target.parents or target in v2.parents:
        raise ProtocolViolation("HPO output overlaps V2 runs")
    if project not in target.parents:
        raise ProtocolViolation("HPO output escaped project root")
    return target


def source_hashes(project_root: Path) -> dict[str, str]:
    """Hash the current source tree, CLI and sealed protocol; never read outcomes."""
    selected = sorted((project_root / "src/neurobridge").rglob("*.py"))
    selected.append(project_root / "tools/phase2a_hpo.py")
    selected.append(project_root / PROTOCOL_RELATIVE)
    return {path.relative_to(project_root).as_posix(): file_sha256(path) for path in selected}


def resolved_config(domain: str, population: str, architecture: str, candidate: Candidate,
                    root_seed: int, channel_indices: Sequence[int] | None = None) -> dict[str, Any]:
    if domain not in DOMAINS or population not in DOMAINS[domain] or architecture not in ARCHITECTURES:
        raise ProtocolViolation("unknown domain/population/architecture")
    effective_seed = root_seed + 1 if domain == "synthetic" and population == "B" else root_seed
    result: dict[str, Any] = {
        "domain": domain, "population": population, "branch": "held_out",
        "architecture": architecture, "objective": candidate.objective,
        "candidate_id": candidate.candidate_id, "candidate_index": candidate.index,
        "learning_rate": float(candidate.lr), "weight_decay": float(candidate.weight_decay),
        "active_temperature": float(candidate.temperature),
        "exact_hyperparameter_strings": {"lr": candidate.lr, "weight_decay": candidate.weight_decay,
                                         "temperature": candidate.temperature},
        "temperature_field": ("embedding_temperature" if candidate.objective in {"soft", "infonce"}
                              else "cebra_temperature"),
        "metadata_temperature": 0.5, "batch_size": 1024, "embedding_dim": 3,
        "window_size": 21, "stride": 1, "positive_offset": 10,
        "optimizer": "AdamW", "schedule": SCHEDULES[domain],
        "training_seed_root": root_seed, "training_seed_effective": effective_seed,
        "split_seed": SPLIT_SEED,
        "architecture_settings": ({"hidden_dim": 64, "cnn_layers": 3}
                                  if architecture == "cnn1d" else
                                  {"transformer_dim": 64, "transformer_heads": 4,
                                   "transformer_layers": 2, "transformer_dropout": 0.1}),
    }
    if domain == "synthetic":
        result.update({"dataset_generator_seed": DATA_SEED, "n_neurons": 160 if population == "A" else 120,
                       "synthetic_observation_lag_bins": 10})
    else:
        if channel_indices is None:
            raise ProtocolViolation("real config requires verified canonical channel indices")
        result.update({"channel_partition_seed": CHANNEL_PARTITION_SEED,
                       "channel_indices": list(channel_indices), "imposed_shift_bins": 0})
    return result


def trial_id(config: Mapping[str, Any]) -> str:
    digest = _sha256_bytes(_canonical_json(config).encode("utf-8"))[:16]
    return (f"{STUDY_ID}-{config['domain']}-{config['population']}-{config['architecture']}-"
            f"{config['objective']}-c{config['candidate_index']}-s{config['training_seed_root']}-{digest}")


def plan_initial(project_root: Path, candidates: Mapping[str, Sequence[Candidate]],
                 channels: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    output_root = guarded_output_root(project_root)
    plan: list[dict[str, Any]] = []
    for domain, populations in DOMAINS.items():
        for objective in OBJECTIVES:
            for architecture in ARCHITECTURES:
                for index in INITIAL_CANDIDATES:
                    for population in populations:
                        config = resolved_config(domain, population, architecture, candidates[objective][index],
                                                 SEARCH_SEED, channels.get(population) if domain == "real" else None)
                        identifier = trial_id(config)
                        destination = output_root / "studies" / STUDY_ID / "trials" / identifier
                        if output_root not in destination.resolve().parents:
                            raise ProtocolViolation("planned trial output escaped HPO root")
                        plan.append({"trial_id": identifier, "candidate_id": config["candidate_id"],
                                     "config": config, "config_sha256": _sha256_bytes(
                                         _canonical_json(config).encode("utf-8")),
                                     "output_path": str(destination),
                                     "safe_input_manifest_planned": str(output_root / "safe_inputs" / domain /
                                                                         population / "manifest.json")})
    if len(plan) != 160 or len({row["trial_id"] for row in plan}) != 160:
        raise ProtocolViolation("initial trial count or IDs differ from frozen budget")
    return plan


def check_safe_input_manifest(manifest: Mapping[str, Any], safe_root: Path) -> None:
    """Validate a future train/validation-only input contract; do not prepare it."""
    allowed = {"domain", "population", "train_trial_ids", "validation_trial_ids", "windows_path",
               "windows_sha256", "split_sha256", "raw_source_sha256", "dataset_sha256",
               "channel_partition_sha256", "valid_mask_sha256", "parent_artifact_id"}
    if set(manifest) - allowed:
        raise ProtocolViolation("safe input manifest has forbidden/unknown keys")
    if not {"domain", "population", "train_trial_ids", "validation_trial_ids", "windows_path",
            "windows_sha256", "split_sha256", "raw_source_sha256", "dataset_sha256"} <= set(manifest):
        raise ProtocolViolation("safe input manifest lacks parent hashes or split IDs")
    if manifest["domain"] not in DOMAINS or manifest["population"] not in DOMAINS[manifest["domain"]]:
        raise ProtocolViolation("safe input domain/population mismatch")
    train, validation = manifest["train_trial_ids"], manifest["validation_trial_ids"]
    if not train or not validation or not all(isinstance(x, int) for x in train + validation):
        raise ProtocolViolation("safe input train/validation trial IDs invalid")
    if set(train) & set(validation):
        raise ProtocolViolation("train/validation trial overlap")
    path = Path(manifest["windows_path"]).resolve()
    if safe_root.resolve() not in path.parents:
        raise ProtocolViolation("safe window data must live under HPO-safe input root")
    if any(part.lower() in {"test", "stage05_metrics"} for part in path.parts):
        raise ProtocolViolation("test/outcome path forbidden")
    for key in ("windows_sha256", "split_sha256", "raw_source_sha256", "dataset_sha256"):
        value = manifest[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ProtocolViolation(f"invalid {key}")


def immutable_manifest(path: Path, payload: Mapping[str, Any], output_root: Path) -> str:
    """Create once, then resume only if byte-identical; stale parents need a new ID."""
    root = output_root.resolve()
    target = path.resolve()
    if root not in target.parents:
        raise ProtocolViolation("manifest write outside HPO output root")
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProtocolViolation("existing immutable manifest differs; new trial/study ID required")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ProtocolViolation("concurrent manifest conflicts with frozen content")
        return "reused"
    return "created"


def validation_geometry(embedding: Any, trial_id_values: Any, time_id_values: Any,
                        valid_mask: Any, split_values: Any) -> dict[str, float | int | bool]:
    """Existing frozen validation-only three-way near-collapse diagnostic."""
    import numpy as np

    z = np.asarray(embedding, dtype=np.float64)
    metadata = [np.asarray(x) for x in (trial_id_values, time_id_values, valid_mask, split_values)]
    if z.ndim != 2 or z.shape[1] != 3 or z.shape[0] < 2:
        raise ProtocolViolation("validation embedding must be N x 3 with N >= 2")
    if any(len(x) != len(z) for x in metadata):
        raise ProtocolViolation("embedding/metadata cardinality mismatch")
    if not np.all(np.asarray(valid_mask, dtype=bool)) or not np.all(np.asarray(split_values) == "validation"):
        raise ProtocolViolation("diagnostic received invalid or non-validation rows")
    if not np.isfinite(z).all():
        raise ProtocolViolation("validation embedding contains nonfinite values")
    norms = np.linalg.norm(z, axis=1)
    if not np.all(np.abs(norms - 1.0) <= 1e-3):
        raise ProtocolViolation("validation embedding is not unit norm")
    sample_count = min(len(z), PAIR_SAMPLE_MAX)
    rng = np.random.default_rng(PAIR_SAMPLE_SEED)
    sample = z[rng.choice(len(z), size=sample_count, replace=False)]
    delta = sample[:, None, :] - sample[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)[np.triu_indices(sample_count, k=1)]
    d50 = float(np.quantile(distances, 0.5))
    f01 = float(np.mean(distances < 0.01))
    eigenvalues = np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1]
    trace = float(np.sum(eigenvalues))
    return {"n_validation_rows": len(z), "pair_sample_rows": sample_count,
            "pair_sampling_seed": PAIR_SAMPLE_SEED, "distance_q50": d50,
            "fraction_distance_lt_0_01": f01, "covariance_trace": trace,
            "near_collapse": bool(d50 < 0.02 and f01 >= 0.25 and trace < 0.001)}


_OUTCOME_KEYS = {"domain", "population", "architecture", "objective", "candidate_index",
                 "training_seed_root", "status", "validation_loss", "selected_update", "stopping_update"}
_ELIGIBLE = "ELIGIBLE"


def validate_outcome(record: Mapping[str, Any]) -> None:
    """Selection accepts a narrow validation-only ledger, never arbitrary metrics."""
    if set(record) != _OUTCOME_KEYS:
        raise ProtocolViolation("outcome has missing/extra fields (test/scientific metrics forbidden)")
    if (record["domain"] not in DOMAINS or record["population"] not in DOMAINS[record["domain"]]
            or record["architecture"] not in ARCHITECTURES or record["objective"] not in OBJECTIVES
            or record["candidate_index"] not in range(6)
            or record["training_seed_root"] not in {SEARCH_SEED, *FINALIST_SEEDS}):
        raise ProtocolViolation("outcome key outside frozen trial plan")
    if record["status"] not in {"ELIGIBLE", "FAILED_RUNTIME", "FAILED_NUMERICAL",
                                "FAILED_ARTIFACT", "INELIGIBLE_NEAR_COLLAPSE"}:
        raise ProtocolViolation("unknown trial status")
    if record["status"] == _ELIGIBLE:
        value = record["validation_loss"]
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ProtocolViolation("eligible trial needs finite validation objective loss")
        if not isinstance(record["selected_update"], int) or record["selected_update"] < 1:
            raise ProtocolViolation("eligible trial needs selected validation update")
        if not isinstance(record["stopping_update"], int) or record["stopping_update"] < 1:
            raise ProtocolViolation("eligible trial needs stopping update")


def _population_records(records: Sequence[Mapping[str, Any]], domain: str, architecture: str,
                        objective: str, candidate_index: int, seed: int) -> list[Mapping[str, Any]]:
    subset = [row for row in records if (row["domain"], row["architecture"], row["objective"],
                row["candidate_index"], row["training_seed_root"]) ==
              (domain, architecture, objective, candidate_index, seed)]
    expected = set(DOMAINS[domain])
    if len(subset) != len(expected) or {row["population"] for row in subset} != expected:
        return []
    return subset


def ranked_search_candidates(records: Sequence[Mapping[str, Any]], domain: str,
                             architecture: str, objective: str,
                             indices: Sequence[int] = INITIAL_CANDIDATES) -> list[tuple[int, float]]:
    for row in records:
        validate_outcome(row)
    ranked = []
    for index in indices:
        subset = _population_records(records, domain, architecture, objective, index, SEARCH_SEED)
        if subset and all(row["status"] == _ELIGIBLE for row in subset):
            ranked.append((index, sum(float(row["validation_loss"]) for row in subset) / len(subset)))
    return sorted(ranked, key=lambda item: (item[1], item[0]))


def extension_required(records: Sequence[Mapping[str, Any]], domain: str, objective: str) -> bool:
    """Paired trigger: either architecture has <2 all-population eligible initial triples."""
    return any(len(ranked_search_candidates(records, domain, architecture, objective)) < 2
               for architecture in ARCHITECTURES)


def finalist_indices(records: Sequence[Mapping[str, Any]], domain: str,
                     architecture: str, objective: str, extended: bool) -> tuple[int, ...]:
    indices = INITIAL_CANDIDATES + EXTENSION_CANDIDATES if extended else INITIAL_CANDIDATES
    return tuple(index for index, _ in ranked_search_candidates(records, domain, architecture,
                                                                 objective, indices)[:2])


def select_finalist(records: Sequence[Mapping[str, Any]], domain: str, architecture: str,
                    objective: str, finalists: Sequence[int]) -> int | None:
    """All 3 HPO seeds and populations qualify; mean loss then frozen tie-breaker."""
    for row in records:
        validate_outcome(row)
    ranking: list[tuple[int, float, int]] = []
    for index in finalists:
        groups = [_population_records(records, domain, architecture, objective, index, seed)
                  for seed in (SEARCH_SEED, *FINALIST_SEEDS)]
        if any(not group or any(row["status"] != _ELIGIBLE for row in group) for group in groups):
            continue
        flat = [row for group in groups for row in group]
        ranking.append((index, sum(float(row["validation_loss"]) for row in flat) / len(flat),
                        sum(int(row["stopping_update"]) for row in flat)))
    if not ranking:
        return None
    best_loss = min(loss for _, loss, _ in ranking)
    tied = [item for item in ranking if item[1] - best_loss <= 1e-6]
    return min(tied, key=lambda item: (item[2], item[0]))[0]


def dry_run(project_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    project = project_root.resolve()
    destination = guarded_output_root(project, output_root)
    candidates, checksum = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    channels, channel_sha = channel_partition(project)
    plan = plan_initial(project, candidates, channels)
    sources = source_hashes(project)
    source_tree_sha = _sha256_bytes(_canonical_json(sources).encode("utf-8"))
    if any("test" in Path(item["safe_input_manifest_planned"]).parts for item in plan):
        raise ProtocolViolation("dry-run references a test input")
    report = {
        "study_id": STUDY_ID, "mode": "dry_run_no_fit_no_optimizer",
        "candidate_sha256": checksum, "channel_partition_sha256": channel_sha,
        "source_file_sha256": sources, "source_tree_sha256": source_tree_sha,
        "initial_trial_count": len(plan), "initial_trials": plan,
        "candidate_zero": {objective: asdict(values[0]) for objective, values in candidates.items()},
        "reserved_candidate_ids": [values[index].candidate_id for values in candidates.values()
                                   for index in EXTENSION_CANDIDATES],
        "future_activation": {
            "extension": "after candidates 0-3 on seed 1101: if either architecture has <2 all-population eligible triples in an objective/domain pair, activate sealed 4-5 for both",
            "finalists": "rank up to 2 eligible triples/cell on validation mean; re-evaluate seeds 1201/1301; require all 3 seeds and populations",
        },
        "parent_data_split_hashes": "PENDING_HPO_SAFE_TRAIN_VALIDATION_INPUTS",
        "hard_firewall": {
            "test_indices_loaded": False, "test_data_loaded": False, "test_artifacts_referenced": False,
            "stage05_or_scientific_outcomes_loaded": False, "v2_output_writable": False,
            "fit_hooks_called": False, "optimizer_steps": 0,
        },
        "fit_readiness": "NO_GO_UNTIL_TRAIN_VALIDATION_ONLY_INPUTS_AND_SAFE_FIT_ADAPTER_EXIST",
    }
    # Dry-run outputs are outside V2.  An existing different report cannot be overwritten.
    immutable_manifest(destination / "dry_run" / "initial_plan.json", report, destination)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Phase-2A HPO planner (no fit entry point)")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    report = dry_run(args.project_root)
    print(_canonical_json({key: value for key, value in report.items() if key != "initial_trials"}))
    print(f"initial_trials={report['initial_trial_count']} plan={guarded_output_root(args.project_root) / 'dry_run/initial_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
