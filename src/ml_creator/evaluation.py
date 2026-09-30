"""Small deterministic, entity-disjoint, time-ordered evaluation harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core import FitGuard, PredictionEvent, artifact_hash


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    return sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))


@dataclass(frozen=True)
class FrozenPortfolio:
    selected_features: tuple[str, ...]
    research_hash: str
    selection_hash: str


def freeze(selected_features: Sequence[str], research_hash: str) -> FrozenPortfolio:
    if not selected_features:
        raise ValueError("NO_CANDIDATES")
    selected = tuple(sorted(set(selected_features)))
    return FrozenPortfolio(selected, research_hash, artifact_hash((selected, research_hash)))


class MeanDifferenceModel:
    """One-feature training-only score, intended as a test baseline."""

    def fit(self, values: Sequence[float], labels: Sequence[int], indices: Sequence[int], guard: FitGuard):
        guard.check(indices)
        pos = [values[i] for i in indices if labels[i] == 1]
        neg = [values[i] for i in indices if labels[i] == 0]
        if not pos or not neg:
            raise ValueError("training fold requires both classes")
        self.direction = 1 if sum(pos) / len(pos) >= sum(neg) / len(neg) else -1
        return self

    def score(self, value: float) -> float:
        return self.direction * value


def time_ordered_oof(events: Sequence[PredictionEvent], values: Sequence[float], *,
                     blocks: int = 4) -> dict:
    """Growing-window OOF; reserves the last chronological block as sealed holdout."""
    if len(events) != len(values) or len(events) < blocks * 2 or blocks < 3:
        raise ValueError("insufficient aligned rows or blocks")
    if len({e.entity_id for e in events}) != len(events):
        raise ValueError("entity leakage: repeated entity")
    order = sorted(range(len(events)), key=lambda i: (events[i].as_of_ts, events[i].entity_id))
    groups = [order[i * len(order) // blocks:(i + 1) * len(order) // blocks] for i in range(blocks)]
    labels = [e.label for e in events]
    predictions: list[tuple[int, float]] = []
    for block in range(1, blocks - 1):
        train = [i for group in groups[:block] for i in group]
        valid = groups[block]
        if max(events[i].label_ts for i in train) > min(events[i].as_of_ts for i in valid):
            raise ValueError("temporal leakage: training labels unavailable at validation time")
        model = MeanDifferenceModel().fit(values, labels, train, FitGuard(train))
        predictions.extend((i, model.score(values[i])) for i in valid)
    oof_auc = auc([labels[i] for i, _ in predictions], [score for _, score in predictions])
    return {"oof_auc": oof_auc, "oof_rows": len(predictions),
            "holdout_indices": tuple(groups[-1]), "status": "FINE"}


def evaluate_sealed_holdout(events: Sequence[PredictionEvent], values: Sequence[float],
                            portfolio: FrozenPortfolio, holdout_indices: Sequence[int]) -> dict:
    if not portfolio.selection_hash or not portfolio.selected_features:
        raise ValueError("portfolio must be frozen before holdout")
    held = set(holdout_indices)
    train = [i for i in range(len(events)) if i not in held]
    if not held or not train or max(events[i].label_ts for i in train) > min(events[i].as_of_ts for i in held):
        raise ValueError("invalid chronological holdout")
    labels = [e.label for e in events]
    model = MeanDifferenceModel().fit(values, labels, train, FitGuard(train))
    indices = sorted(held)
    return {"status": "HOLDOUT_EVALUATED", "portfolio_hash": portfolio.selection_hash,
            "holdout_auc": auc([labels[i] for i in indices], [model.score(values[i]) for i in indices])}
