"""Run no-data research from a per-run JSON configuration."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .core import DataState, Problem, RunIntent, RunMode, artifact_hash, route
from .research import (FeatureResearchRun, StructureResearchRun, normalize_source_types,
                       research_features, research_structures)


def write_json_atomic(path: Path, value: dict) -> None:
    """Keep the previous artifact intact if serialization or writing fails."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run_config(config: dict, *, mode_override: str | None = None,
               problem_override: str | None = None, entity_override: str | None = None,
               horizon_override: int | None = None, no_horizon: bool = False,
               feature_count_override: int | None = None, structure_count_override: int | None = None,
               sources_override: str | None = None,
               on_feature_round: Callable[[Problem, RunIntent, FeatureResearchRun], None] | None = None,
               on_structure_round: Callable[[Problem, RunIntent, StructureResearchRun], None] | None = None) -> dict:
    if not isinstance(config, dict) or not all(isinstance(config.get(name), dict) for name in ("problem", "intent")):
        raise ValueError("config must contain problem and intent objects")
    if not isinstance(config.get("research", {}), dict):
        raise ValueError("research must be an object")
    intent_data = config["intent"]
    intent = RunIntent(RunMode(mode_override or intent_data["mode"]), DataState(intent_data["data_state"]))
    if intent.data_state != DataState.NO_DATA:
        raise ValueError("DATA_AVAILABLE requires a dataset adapter, which is not implemented in this CLI")
    problem_data = dict(config["problem"])
    if problem_override is not None:
        problem_data["task"] = problem_override
        if horizon_override is None:
            # A new problem must not inherit a different problem's horizon.
            problem_data.pop("horizon_days", None)
    if entity_override is not None:
        problem_data["entity"] = entity_override
    if no_horizon:
        if horizon_override is not None:
            raise ValueError("Choose --horizon-days or --no-horizon, not both")
        problem_data.pop("horizon_days", None)
    if horizon_override is not None:
        problem_data["horizon_days"] = horizon_override
    if problem_data.get("horizon_days") == "no-horizon":
        problem_data["horizon_days"] = None
    try:
        problem = Problem(**problem_data)
    except TypeError as exc:
        raise ValueError(f"invalid problem fields: {exc}") from exc
    limits = config.get("research", {})
    source_types = normalize_source_types(sources_override if sources_override is not None
                                          else limits.get("sources", ["KAGGLE", "PAPERS", "REPOS"]))
    feature_count = int(feature_count_override if feature_count_override is not None
                        else limits.get("requested_features", 5))
    structure_count = int(structure_count_override if structure_count_override is not None
                          else limits.get("requested_model_structures", 3))
    if feature_count < 0 or structure_count < 0:
        raise ValueError("requested research counts cannot be negative")
    if intent.mode == RunMode.FEATURES and feature_count == 0:
        raise ValueError("FEATURES mode requires requested_features > 0 in the config")
    if intent.mode == RunMode.STRUCTURES and structure_count == 0:
        raise ValueError("STRUCTURES mode requires requested_model_structures > 0 in the config")
    feature_callback = ((lambda run: on_feature_round(problem, intent, run))
                        if on_feature_round else None)
    feature_run = (research_features(problem, count=feature_count,
                                    max_rounds=int(limits.get("max_feature_rounds", 5)),
                                    source_types=source_types,
                                    on_round=feature_callback)
                   if intent.mode in (RunMode.FEATURES, RunMode.BOTH) and feature_count else None)
    features = feature_run.proposals if feature_run else ()
    round_callback = ((lambda run: on_structure_round(problem, intent, run))
                      if on_structure_round else None)
    structure_run = (research_structures(problem, count=structure_count,
                                        max_rounds=int(limits["max_rounds"]) if "max_rounds" in limits else None,
                                        source_types=source_types,
                                        on_round=round_callback)
                     if intent.mode in (RunMode.STRUCTURES, RunMode.BOTH) and structure_count else None)
    structures = structure_run.proposals if structure_run else ()
    result = route(intent, problem, features=features, structures=structures)
    if feature_run:
        result["feature_agent"] = {"requested": feature_run.requested,
                                   "returned": len(feature_run.proposals),
                                   "stop_reason": feature_run.stop_reason,
                                   "rounds": feature_run.rounds}
        if feature_run.stop_reason != "TARGET_REACHED":
            result["status"] = feature_run.stop_reason
    if structure_run:
        result["structure_agent"] = {
            "requested": structure_run.requested,
            "returned": len(structure_run.proposals),
            "stop_reason": structure_run.stop_reason,
            "rounds": structure_run.rounds,
        }
        if structure_run.stop_reason != "TARGET_REACHED":
            result["status"] = structure_run.stop_reason
    result["source_evidence_level"] = "SNIPPET_ONLY"
    result["research_sources"] = source_types
    result["research_artifact_hash"] = artifact_hash({"features": features, "structures": structures})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-linked, no-data ML research proposals")
    parser.add_argument("--config", type=Path, default=Path("run.json"))
    parser.add_argument("--mode", choices=[m.value for m in RunMode], help="override config for this run")
    parser.add_argument("--problem", help="prediction problem for this run; clears the configured horizon unless supplied")
    parser.add_argument("--entity", help="prediction entity for this run")
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument("--horizon-days", type=int, help="prediction horizon in days")
    horizon_group.add_argument("--no-horizon", action="store_true", help="clear the horizon in the config")
    parser.add_argument("--requested-features", type=int, help="override feature count for this run")
    parser.add_argument("--requested-model-structures", type=int, help="override structure count for this run")
    parser.add_argument("--sources", help="comma-separated selection: kaggle,papers,repos")
    parser.add_argument("--output", type=Path, help="save frozen JSON research artifact")
    args = parser.parse_args()
    try:
        checkpoint = args.output.with_name(args.output.name + ".checkpoint.json") if args.output else None
        latest_features: FeatureResearchRun | None = None
        latest_structures: StructureResearchRun | None = None
        selected_sources = normalize_source_types(args.sources if args.sources is not None
                                                  else json.loads(args.config.read_text()).get("research", {}).get(
                                                      "sources", ["KAGGLE", "PAPERS", "REPOS"]))

        def write_checkpoint(problem: Problem, intent: RunIntent) -> None:
            if not checkpoint:
                return
            partial = route(intent, problem,
                            features=latest_features.proposals if latest_features else (),
                            structures=latest_structures.proposals if latest_structures else ())
            for name, run in (("feature_agent", latest_features), ("structure_agent", latest_structures)):
                if run:
                    partial[name] = {"requested": run.requested, "returned": len(run.proposals),
                                     "stop_reason": run.stop_reason, "rounds": run.rounds}
            partial["source_evidence_level"] = "SNIPPET_ONLY"
            partial["research_sources"] = selected_sources
            partial["research_artifact_hash"] = artifact_hash({
                "features": latest_features.proposals if latest_features else (),
                "structures": latest_structures.proposals if latest_structures else ()})
            write_json_atomic(checkpoint, partial)

        def save_feature_round(problem: Problem, intent: RunIntent, run: FeatureResearchRun) -> None:
            nonlocal latest_features
            latest_features = run
            write_checkpoint(problem, intent)
            print(f"Feature round {len(run.rounds)}: {len(run.proposals)}/{run.requested} proposals", file=sys.stderr)

        def save_round(problem: Problem, intent: RunIntent, run: StructureResearchRun) -> None:
            nonlocal latest_structures
            latest_structures = run
            write_checkpoint(problem, intent)
            print(f"Structure round {len(run.rounds)}: {len(run.proposals)}/{run.requested} proposals", file=sys.stderr)

        result = run_config(json.loads(args.config.read_text()), mode_override=args.mode,
                            problem_override=args.problem, entity_override=args.entity,
                            horizon_override=args.horizon_days, no_horizon=args.no_horizon,
                            feature_count_override=args.requested_features,
                            structure_count_override=args.requested_model_structures,
                            sources_override=args.sources,
                            on_feature_round=save_feature_round, on_structure_round=save_round)
        if args.output:
            write_json_atomic(args.output, result)
            if checkpoint and checkpoint.exists():
                checkpoint.unlink()
            print(f"Saved {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
