from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable


class RunMode(str, Enum):
    FEATURES = "FEATURES"
    STRUCTURES = "STRUCTURES"
    BOTH = "BOTH"


class DataState(str, Enum):
    DATA_AVAILABLE = "DATA_AVAILABLE"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class RunIntent:
    mode: RunMode
    data_state: DataState


@dataclass(frozen=True)
class Problem:
    task: str
    entity: str
    horizon_days: int | None = None

    def __post_init__(self):
        if not self.task or not self.entity:
            raise ValueError("task and entity are required")
        if self.horizon_days is not None:
            if type(self.horizon_days) is not int or self.horizon_days <= 0:
                raise ValueError("horizon_days must be a positive integer, null, or omitted")


@dataclass(frozen=True)
class ResearchFeature:
    concept_id: str
    name: str
    concept_family: str
    mechanism: str
    required_information: tuple[str, ...]
    typical_formula: str
    source_ids: tuple[str, ...]
    similar_problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelStructure:
    name: str
    problem_type: str
    architecture: str
    components: tuple[str, ...]
    rationale: str
    assumptions: tuple[str, ...]
    required_data_properties: tuple[str, ...]
    training_protocol: str
    validation_protocol: str
    strengths: tuple[str, ...]
    risks: tuple[str, ...]
    sources: tuple[str, ...]
    canonical_structure: str = ""
    reviewer_reason: str = ""


@dataclass(frozen=True)
class PredictionEvent:
    entity_id: str
    as_of_ts: datetime
    label: int
    label_ts: datetime
    horizon_days: int

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError("label must be binary")
        if self.horizon_days <= 0 or not self.entity_id:
            raise ValueError("entity and positive horizon required")
        if not self.as_of_ts < self.label_ts <= self.as_of_ts + timedelta(days=self.horizon_days):
            raise ValueError("label timestamp must follow prediction and fall within horizon")


@dataclass(frozen=True)
class Fact:
    entity_id: str
    valid_ts: datetime
    knowledge_ts: datetime
    amount: float


@dataclass(frozen=True)
class FeatureSpec:
    concept_id: str
    entity_column: str
    valid_ts_column: str
    knowledge_ts_column: str
    value_column: str
    aggregation: str
    lookback_days: int
    grammar_version: str = "v1"


SCHEMA = {"entity_id": "str", "valid_ts": "datetime", "knowledge_ts": "datetime", "amount": "float"}


def compile_feature(concept: ResearchFeature, *, lookback_days: int = 30,
                    aggregation: str = "count", join_path: tuple[str, ...] = ("transactions",)) -> FeatureSpec:
    """Approve only the single-table, typed grammar implemented in this milestone."""
    if not concept.source_ids or "UNVERIFIED_SOURCE" in concept.source_ids:
        raise ValueError("VETOED: unverified or missing source")
    if join_path != ("transactions",):
        raise ValueError("VETOED: unsupported join path")
    if aggregation not in {"count", "sum", "mean"} or lookback_days <= 0:
        raise ValueError("VETOED: unsupported aggregation or lookback")
    if not all(k in SCHEMA for k in ("entity_id", "valid_ts", "knowledge_ts", "amount")):
        raise ValueError("VETOED: invalid schema")
    return FeatureSpec(concept.concept_id, "entity_id", "valid_ts", "knowledge_ts", "amount", aggregation, lookback_days)


def eligible_fact(fact: Fact, event: PredictionEvent, lookback_days: int) -> bool:
    return (fact.entity_id == event.entity_id
            and event.as_of_ts - timedelta(days=lookback_days) < fact.valid_ts <= event.as_of_ts
            and fact.knowledge_ts <= event.as_of_ts)


def generate_variants(concept: ResearchFeature, max_variants: int = 30) -> tuple[FeatureSpec, ...]:
    if not 0 < max_variants <= 30:
        raise ValueError("max_variants must be 1..30")
    return tuple(compile_feature(concept, lookback_days=d, aggregation=a)
                 for d in (7, 30, 90) for a in ("count", "sum", "mean"))[:max_variants]


def calculate(spec: FeatureSpec, event: PredictionEvent, facts: Iterable[Fact]) -> float:
    values = [f.amount for f in facts if eligible_fact(f, event, spec.lookback_days)]
    if spec.aggregation == "count":
        return float(len(values))
    if not values:
        return 0.0
    if spec.aggregation == "sum":
        return sum(values)
    if spec.aggregation == "mean":
        return sum(values) / len(values)
    raise ValueError("VETOED: unsupported aggregation")


class FitGuard:
    """Checks row provenance at fit time, including within fold transforms."""

    def __init__(self, allowed_indices: Iterable[int]):
        self.allowed = frozenset(allowed_indices)

    def check(self, indices: Iterable[int]) -> None:
        if not set(indices) <= self.allowed:
            raise ValueError("VETOED: fit touched validation or holdout rows")


def artifact_hash(value: object) -> str:
    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return asdict(obj)
    payload = json.dumps(value, default=default, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def route(intent: RunIntent, problem: Problem, *, features: tuple[ResearchFeature, ...] = (),
          structures: tuple[ModelStructure, ...] = (), events: tuple[PredictionEvent, ...] = (),
          facts: tuple[Fact, ...] = ()) -> dict:
    """Route frozen research artifacts; never invent evidence or run a seeker."""
    wants_features = intent.mode in (RunMode.FEATURES, RunMode.BOTH)
    wants_structures = intent.mode in (RunMode.STRUCTURES, RunMode.BOTH)
    result = {"intent": asdict(intent), "problem": asdict(problem), "status": "NO_DATA"}
    if wants_features:
        result["feature_research"] = [dict(asdict(f), status="RESEARCH_PROPOSAL") for f in features]
    if wants_structures:
        result["structure_research"] = [dict(asdict(s), status="RESEARCH_PROPOSAL") for s in structures]
    if intent.data_state == DataState.NO_DATA:
        result["data_gaps"] = sorted({need for f in features for need in f.required_information}) if wants_features else []
        return result
    if not events:
        raise ValueError("DATA_AVAILABLE requires prediction events")
    if problem.horizon_days is None:
        raise ValueError("DATA_AVAILABLE requires horizon_days")
    if len({e.entity_id for e in events}) != len(events):
        raise ValueError("MVP permits one prediction event per entity")
    if any(e.horizon_days != problem.horizon_days for e in events):
        raise ValueError("event horizon differs from problem")
    data_max_ts = max((f.knowledge_ts for f in facts), default=None)
    if data_max_ts is None or any(e.as_of_ts + timedelta(days=e.horizon_days) > data_max_ts for e in events):
        raise ValueError("prediction events lack observable outcome horizon")
    result["status"] = "RESEARCH_PROPOSAL"
    if wants_features:
        result["compiled_feature_specs"] = []
        for feature in features:
            try:
                spec = compile_feature(feature)
                result["compiled_feature_specs"].append(asdict(spec))
            except ValueError as exc:
                result.setdefault("vetoes", []).append({"concept_id": feature.concept_id, "reason": str(exc)})
        if not result["compiled_feature_specs"] and intent.mode == RunMode.FEATURES:
            result["status"] = "NO_CANDIDATES"
    if wants_structures:
        result["model_applicability"] = [{"name": s.name, "status": "UNASSESSED", "reason": "model mapper and experiments pending"} for s in structures]
    result["research_artifact_hash"] = artifact_hash({"features": features, "structures": structures})
    return result
