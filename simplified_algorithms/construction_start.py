"""Simplified construction-start detector.

This is the decision core distilled from the ACD/BASC construction work:

* compare current Sentinel-2 NDBI/SWIR summaries to a usable baseline;
* mark a period as changed when the built-up signal jumps;
* walk periods forward causally, so only evidence available by that period is
  used to emit provisional or confirmed starts.

The module expects already-fetched summary values. It does not call CDSE, read
site inventories, store chips, or look at future periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import cos, isfinite, pi
from typing import Iterable, Sequence


NDBI_HARD_THRESHOLD = 0.04
NDBI_SOFT_THRESHOLD = 0.02
SWIR_THRESHOLD = 0.03
MIN_VALID_FRAC = 0.40

INSUFFICIENT = "insufficient_evidence"
NO_CHANGE = "no_change"
PROVISIONAL = "provisional_start"
CONFIRMED = "confirmed_start"


@dataclass(frozen=True)
class IndexStats:
    """One period of already-aggregated Sentinel-2 values for a site."""

    period: str
    ndbi: float
    swir: float
    ndvi: float | None = None
    valid_frac: float = 1.0
    baseline_period: str | None = None


@dataclass(frozen=True)
class ChangeCall:
    period: str
    status: str
    changed: bool | None
    ndbi_delta: float | None = None
    swir_delta: float | None = None
    ndvi_delta: float | None = None
    baseline_period: str | None = None
    reason: str = ""


@dataclass
class CausalCall:
    period: str
    state: str
    changed_now: bool
    n_clear_seen: int
    run_len: int
    provisional_period: str | None = None
    confirmed_period: str | None = None
    notes: list[str] = field(default_factory=list)


def half_month_period(day: date | datetime) -> str:
    """Return the `YYYY-MM-H1` or `YYYY-MM-H2` bucket for a calendar date."""

    if isinstance(day, datetime):
        day = day.date()
    half = "H1" if day.day <= 15 else "H2"
    return f"{day:%Y-%m}-{half}"


def previous_period(period: str) -> str:
    """Return the previous half-month period label."""

    year_s, month_s, half = period.split("-")
    year, month = int(year_s), int(month_s)
    if half == "H2":
        return f"{year:04d}-{month:02d}-H1"
    first = date(year, month, 1)
    prev = first - timedelta(days=1)
    return f"{prev:%Y-%m}-H2"


def square_bbox(lon: float, lat: float, side_m: float = 750.0) -> tuple[float, float, float, float]:
    """Approximate a ground-square bbox in lon/lat degrees.

    This mirrors the simple TypeScript ACD approach. Use a projected CRS when
    exact geometry matters.
    """

    half = side_m / 2.0
    dlat = half / 111_111.0
    dlon = half / (111_111.0 * cos(lat * pi / 180.0))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _finite(*values: float | None) -> bool:
    return all(v is not None and isfinite(v) for v in values)


def compare_to_baseline(current: IndexStats, baseline: IndexStats | None) -> ChangeCall:
    """Compare one current period to one prior usable baseline.

    The output status is nullable in the same spirit as the production spec:
    not assessed is not the same thing as no change.
    """

    if current.valid_frac < MIN_VALID_FRAC:
        return ChangeCall(
            period=current.period,
            status="no_data",
            changed=None,
            reason=f"valid_frac {current.valid_frac:.2f} below {MIN_VALID_FRAC:.2f}",
        )
    if not _finite(current.ndbi, current.swir):
        return ChangeCall(
            period=current.period,
            status="no_data",
            changed=None,
            reason="current NDBI/SWIR is missing or non-finite",
        )
    if baseline is None:
        return ChangeCall(
            period=current.period,
            status="no_prior_baseline",
            changed=None,
            reason="no usable prior period",
        )
    if baseline.valid_frac < MIN_VALID_FRAC or not _finite(baseline.ndbi, baseline.swir):
        return ChangeCall(
            period=current.period,
            status="no_prior_baseline",
            changed=None,
            baseline_period=baseline.period,
            reason="baseline is unusable",
        )

    d_ndbi = current.ndbi - baseline.ndbi
    d_swir = current.swir - baseline.swir
    d_ndvi = None
    if current.ndvi is not None and baseline.ndvi is not None:
        d_ndvi = current.ndvi - baseline.ndvi
    changed = d_ndbi >= NDBI_HARD_THRESHOLD or (
        d_ndbi >= NDBI_SOFT_THRESHOLD and d_swir >= SWIR_THRESHOLD
    )
    return ChangeCall(
        period=current.period,
        status="ok",
        changed=changed,
        ndbi_delta=round(d_ndbi, 4),
        swir_delta=round(d_swir, 4),
        ndvi_delta=round(d_ndvi, 4) if d_ndvi is not None else None,
        baseline_period=current.baseline_period or baseline.period,
        reason="built-up signal increased" if changed else "below change thresholds",
    )


def compare_series(series: Sequence[IndexStats]) -> list[ChangeCall]:
    """Compare each period to the latest prior `ok` period.

    This is the simple stored-baseline strategy from the TypeScript ACD: the
    next period compares to the last usable prior result, not to a future value.
    """

    calls: list[ChangeCall] = []
    baseline: IndexStats | None = None
    for current in series:
        call = compare_to_baseline(current, baseline)
        calls.append(call)
        if call.status == "ok" or (
            call.status == "no_prior_baseline"
            and current.valid_frac >= MIN_VALID_FRAC
            and _finite(current.ndbi, current.swir)
        ):
            baseline = current
    return calls


def causal_onset(
    observations: Iterable[ChangeCall],
    *,
    confirm_n: int = 2,
    min_clear_history: int = 3,
    allow_gap: int = 1,
) -> list[CausalCall]:
    """Turn per-period change calls into provisional/confirmed onset states."""

    out: list[CausalCall] = []
    n_clear = 0
    run_len = 0
    misses = 0
    provisional: str | None = None
    confirmed: str | None = None

    for obs in observations:
        clear = obs.changed is not None
        if not clear:
            call = CausalCall(
                period=obs.period,
                state=INSUFFICIENT,
                changed_now=False,
                n_clear_seen=n_clear,
                run_len=run_len,
                provisional_period=provisional,
                confirmed_period=confirmed,
                notes=[obs.reason or obs.status],
            )
            out.append(call)
            continue

        n_clear += 1
        changed = bool(obs.changed)
        if changed:
            run_len += 1
            misses = 0
        else:
            misses += 1
            if misses > allow_gap:
                run_len = 0
                if confirmed is None:
                    provisional = None

        if n_clear < min_clear_history:
            state = INSUFFICIENT
            notes = [f"only {n_clear} clear look(s); need {min_clear_history}"]
        elif run_len == 0:
            state = NO_CHANGE
            notes = []
        elif run_len >= confirm_n:
            if provisional is None:
                provisional = obs.period
            if confirmed is None:
                confirmed = provisional
            state = CONFIRMED
            notes = [f"confirmed from first changed look {confirmed}"]
        else:
            if provisional is None:
                provisional = obs.period
            state = PROVISIONAL
            notes = [f"changed at {run_len}/{confirm_n} clear looks"]

        out.append(
            CausalCall(
                period=obs.period,
                state=state,
                changed_now=changed,
                n_clear_seen=n_clear,
                run_len=run_len,
                provisional_period=provisional,
                confirmed_period=confirmed,
                notes=notes,
            )
        )
    return out


def _demo() -> None:
    series = [
        IndexStats("2026-01-H1", ndbi=0.08, swir=0.20, ndvi=0.35),
        IndexStats("2026-01-H2", ndbi=0.081, swir=0.205, ndvi=0.34),
        IndexStats("2026-02-H1", ndbi=0.079, swir=0.203, ndvi=0.36),
        IndexStats("2026-02-H2", ndbi=0.105, swir=0.240, ndvi=0.28),
        IndexStats("2026-03-H1", ndbi=0.145, swir=0.275, ndvi=0.19),
    ]
    changes = compare_series(series)
    calls = causal_onset(changes)
    for change, call in zip(changes, calls):
        print(
            f"{change.period}: changed={change.changed!s:<5} "
            f"d_ndbi={change.ndbi_delta!s:<6} state={call.state}"
        )


if __name__ == "__main__":
    _demo()
