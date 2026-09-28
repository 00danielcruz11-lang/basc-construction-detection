# Simplified Construction And Thermal Algorithms

These are compact, readable versions of the construction and thermal research
algorithms found on this machine.

They deliberately skip data fetching, authentication, dashboards, imagery chips,
and project-specific inventory joins. Feed them already-summarized rows and they
return the same core decisions the larger scripts are built around.

## 1. Construction Start

Source material:

- `~/Aterio/Computer-visual-model/ACD_ALGORITHM.md`
- `basc/HOW_IT_WORKS.txt`
- `basc/detect.py`

Core idea:

1. Split time into half-month periods.
2. For each site and period, compare current Sentinel-2 summary values against
   the latest usable baseline.
3. Call a change when built-up signal rises:

   ```text
   ndbi_delta >= 0.04
   or
   ndbi_delta >= 0.02 and swir_delta >= 0.03
   ```

4. Walk periods forward causally:

   - no clear observation means insufficient evidence, not no change
   - one changed clear look is a provisional start
   - two changed clear looks confirm the start, dated to the first changed look

Run a tiny demo:

```bash
python simplified_algorithms/construction_start.py
```

Use it from Python:

```python
from simplified_algorithms.construction_start import IndexStats, compare_to_baseline

cur = IndexStats(period="2026-03-H1", ndbi=0.13, swir=0.24, ndvi=0.18)
base = IndexStats(period="2026-02-H2", ndbi=0.08, swir=0.20, ndvi=0.26)
print(compare_to_baseline(cur, base).changed)
```

## 2. Thermal Anomaly

Source material:

- `~/Aterio/landsat-thermal/refetch16.py`
- `~/Aterio/landsat-thermal/reanalyze.py`
- `~/Aterio/landsat-thermal/commonmode.py`

Core idea:

1. Fetch Landsat brightness temperature with QA masking upstream.
2. Keep sufficiently clear building boxes.
3. On each date, subtract the median temperature of other buildings on the same
   campus. This removes common weather and scene effects.
4. For each building, scan for one sustained step change in the anomaly series.
5. Treat a positive, significant step as a thermal rise candidate.

Run a tiny demo:

```bash
python simplified_algorithms/thermal_anomaly.py
```

The module uses a deterministic permutation test by default, so repeated runs
produce the same result.

## Why This Folder Exists

The production/research folders have the full plumbing. This folder is the
auditable algorithm skeleton: small enough to inspect in one sitting, but close
enough to the original scripts to preserve the actual decision logic.
