"""Regression gate for a harness report.

``baselines.json`` holds floors under ``min`` and ceilings under ``max``.
A metric fails when it is missing, null, below its floor, or above its
ceiling. Metrics that are not listed are not gated. The note field is
documentation and is ignored by the check.

The checked-in floors are the Phase 3 and Phase 5 scores on the starter
questions: retrieval hit and recall were 1, and routing accuracy was 1
with no false or missed fallback. Generation floors stay unset until a
harness run on the grown set records them.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag.core.config import PROJECT_ROOT

DEFAULT_BASELINES = PROJECT_ROOT / "data" / "evaluation" / "baselines.json"

_TOP = frozenset({"min", "max", "note"})


@dataclass(frozen=True)
class GateFailure:
    """One metric that missed its bound."""

    metric: str
    actual: float | None
    bound: float
    direction: Literal["min", "max"]


def load_baselines(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Read floors and ceilings. A bad file names the field."""

    if not path.is_file():
        raise FileNotFoundError(f"baseline file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    extra = sorted(set(raw) - _TOP)
    if extra:
        raise ValueError(f"{path.name} has unknown fields: {extra}")
    if "note" in raw and not isinstance(raw["note"], str):
        raise ValueError(f"{path.name} note must be a string")
    floors = _bounds(raw.get("min", {}), file_name=path.name, label="min")
    ceilings = _bounds(raw.get("max", {}), file_name=path.name, label="max")
    return floors, ceilings


def check_gate(
    metrics: Mapping[str, float | None],
    floors: Mapping[str, float],
    ceilings: Mapping[str, float],
) -> tuple[GateFailure, ...]:
    """Return one failure per bound the report missed. Empty means pass."""

    failures: list[GateFailure] = []
    for metric, floor in floors.items():
        actual = _actual(metrics, metric)
        if actual is None or actual < floor:
            failures.append(GateFailure(metric=metric, actual=actual, bound=floor, direction="min"))
    for metric, ceiling in ceilings.items():
        actual = _actual(metrics, metric)
        if actual is None or actual > ceiling:
            failures.append(
                GateFailure(metric=metric, actual=actual, bound=ceiling, direction="max")
            )
    return tuple(failures)


def _bounds(raw: object, *, file_name: str, label: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{file_name} {label} must be an object")
    bounds: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{file_name} {label} has an empty metric name")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{file_name} {label}.{key} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{file_name} {label}.{key} must be a finite number")
        bounds[key] = number
    return bounds


def _actual(metrics: Mapping[str, float | None], metric: str) -> float | None:
    if metric not in metrics:
        return None
    value = metrics[metric]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{metric} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{metric} must be a finite number")
    return number
