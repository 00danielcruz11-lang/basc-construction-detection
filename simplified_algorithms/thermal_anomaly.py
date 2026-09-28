"""Simplified Landsat thermal anomaly detector.

This distills the Landsat thermal research scripts into a small reusable core:

* accept already-fetched, QA-masked brightness-temperature observations;
* compare each building to the same-day leave-one-out campus median;
* find the strongest sustained step in that anomaly series;
* call positive significant steps thermal-rise candidates.

The fetch scripts that inspired this used Landsat B10 brightness temperature,
masked BQA bits for cloud/cirrus/shadow, and kept boxes with at least 75% clear
pixels. This module starts after that fetch step.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import mean, median, pstdev
from typing import Iterable


MIN_PEERS_PER_DATE = 12
MIN_SEGMENT = 8
DEFAULT_PERMUTATIONS = 1000


@dataclass(frozen=True)
class ThermalObservation:
    building: str
    date: str
    bt_k: float
    stage: str = ""
    clear_pixels: int | None = None
    total_pixels: int | None = None


@dataclass(frozen=True)
class ThermalAnomaly:
    building: str
    date: str
    anomaly_c: float
    bt_c: float
    reference_c: float
    n_peers: int
    stage: str = ""


@dataclass(frozen=True)
class StepResult:
    building: str
    onset_date: str | None
    step_c: float | None
    t_stat: float | None
    p_value: float | None
    n_before: int
    n_after: int
    verdict: str


def keep_clear(obs: ThermalObservation, min_clear_frac: float = 0.75) -> bool:
    """Return whether an observation passes the clear-pixel gate."""

    if obs.clear_pixels is None or obs.total_pixels is None:
        return True
    if obs.total_pixels <= 0:
        return False
    return obs.clear_pixels / obs.total_pixels >= min_clear_frac


def campus_anomalies(
    observations: Iterable[ThermalObservation],
    *,
    min_peers: int = MIN_PEERS_PER_DATE,
    min_clear_frac: float = 0.75,
) -> list[ThermalAnomaly]:
    """Subtract the same-day leave-one-out campus median from each building."""

    by_date: dict[str, list[ThermalObservation]] = {}
    for obs in observations:
        if keep_clear(obs, min_clear_frac):
            by_date.setdefault(obs.date, []).append(obs)

    anomalies: list[ThermalAnomaly] = []
    for day, rows in sorted(by_date.items()):
        if len(rows) < min_peers:
            continue
        c_by_building = {row.building: row.bt_k - 273.15 for row in rows}
        for row in rows:
            others = [v for building, v in c_by_building.items() if building != row.building]
            if len(others) < min_peers - 1:
                continue
            ref = median(others)
            bt_c = c_by_building[row.building]
            anomalies.append(
                ThermalAnomaly(
                    building=row.building,
                    date=day,
                    anomaly_c=bt_c - ref,
                    bt_c=bt_c,
                    reference_c=ref,
                    n_peers=len(others),
                    stage=row.stage,
                )
            )
    return anomalies


def _step_stat(values: list[float], min_segment: int) -> tuple[float | None, int | None]:
    best_t: float | None = None
    best_i: int | None = None
    n = len(values)
    for i in range(min_segment, n - min_segment + 1):
        before = values[:i]
        after = values[i:]
        before_mean = mean(before)
        after_mean = mean(after)
        se = sqrt((pstdev(before) ** 2 / len(before)) + (pstdev(after) ** 2 / len(after)))
        if se <= 0:
            if before_mean == after_mean:
                continue
            t_stat = 1_000_000_000.0 if after_mean > before_mean else -1_000_000_000.0
        else:
            t_stat = (after_mean - before_mean) / se
        if best_t is None or abs(t_stat) > abs(best_t):
            best_t = t_stat
            best_i = i
    return best_t, best_i


def best_step(
    series: list[ThermalAnomaly],
    *,
    min_segment: int = MIN_SEGMENT,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 7,
) -> StepResult:
    """Find one sustained step change in a building's anomaly series."""

    if not series:
        return StepResult("", None, None, None, None, 0, 0, "no data")

    ordered = sorted(series, key=lambda row: row.date)
    building = ordered[0].building
    values = [row.anomaly_c for row in ordered]
    if len(values) < min_segment * 2:
        return StepResult(building, None, None, None, None, len(values), 0, "too few points")

    t_stat, index = _step_stat(values, min_segment)
    if t_stat is None or index is None:
        return StepResult(building, None, None, None, None, len(values), 0, "no variable signal")

    before = values[:index]
    after = values[index:]
    step = mean(after) - mean(before)

    rng = Random(seed)
    null_hits = 0
    for _ in range(permutations):
        shuffled = values[:]
        rng.shuffle(shuffled)
        null_t, _ = _step_stat(shuffled, min_segment)
        if null_t is not None and abs(null_t) >= abs(t_stat):
            null_hits += 1
    p_value = null_hits / permutations if permutations else None

    if p_value is not None and p_value < 0.05 and step > 0:
        verdict = "thermal_rise"
    elif p_value is not None and p_value < 0.05 and step < 0:
        verdict = "thermal_drop"
    else:
        verdict = "no_significant_step"

    return StepResult(
        building=building,
        onset_date=ordered[index].date,
        step_c=round(step, 3),
        t_stat=round(t_stat, 3),
        p_value=round(p_value, 4) if p_value is not None else None,
        n_before=len(before),
        n_after=len(after),
        verdict=verdict,
    )


def detect_thermal_steps(
    observations: Iterable[ThermalObservation],
    *,
    min_peers: int = MIN_PEERS_PER_DATE,
    min_segment: int = MIN_SEGMENT,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 7,
) -> list[StepResult]:
    """Run the full simplified thermal algorithm for every building."""

    anomalies = campus_anomalies(observations, min_peers=min_peers)
    by_building: dict[str, list[ThermalAnomaly]] = {}
    for row in anomalies:
        by_building.setdefault(row.building, []).append(row)
    return [
        best_step(rows, min_segment=min_segment, permutations=permutations, seed=seed)
        for _, rows in sorted(by_building.items())
    ]


def _demo_observations() -> list[ThermalObservation]:
    rows: list[ThermalObservation] = []
    for day in range(1, 25):
        date = f"2026-01-{day:02d}"
        for building_num in range(1, 15):
            building = f"B{building_num:02d}"
            baseline = 303.0 + (building_num % 3) * 0.15
            if building == "B13" and day >= 14:
                baseline += 2.2
            rows.append(ThermalObservation(building, date, baseline, clear_pixels=16, total_pixels=16))
    return rows


def _demo() -> None:
    for result in detect_thermal_steps(
        _demo_observations(),
        min_peers=10,
        min_segment=6,
        permutations=300,
    ):
        if result.verdict in {"thermal_rise", "thermal_drop"}:
            print(result)


if __name__ == "__main__":
    _demo()
