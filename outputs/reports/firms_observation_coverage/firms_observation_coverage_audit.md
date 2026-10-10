# FIRMS observation coverage audit — frozen outputs preserved

## Findings

Study dates: 2016-09-22–2026-08-31. Frozen cohort: 46 grids. Raw archive/NRT rows are counted before spatial/confidence/date filtering; retained counts use the exact existing mainland polygon, extent and normal/high confidence filter. Raw CSVs contain detection records only, not observation footprints or per-grid coverage certificates.

| window | days | raw_records | retained_sumatra_records | zero_raw_days |
| --- | --- | --- | --- | --- |
| focus_2022_gap | 15 | 0 | 0 | 15 |
| focus_2026_may | 31 | 0 | 0 | 31 |
| focus_2026_july | 6 | 0 | 0 | 6 |

Across the study period, **84 days have no raw records anywhere in the supplied files**, and 211 days have no retained Sumatra detections. These are different conditions. Neither an empty detection table nor record presence establishes complete observable grid coverage. May 2026 has no automatic sensor-outage explanation: NASA reports S-NPP safe from May 1 at 05:00 UTC. Its local gap requires download/distribution/archive provenance investigation; the cause is unresolved.

## Impact on the frozen cohort

| window | split | origin_rows | target_rows | either_rows | target_no_activity_rows |
| --- | --- | --- | --- | --- | --- |
| focus_2022_gap | test | 0 | 0 | 0 | 0 |
| focus_2022_gap | train | 690 | 690 | 736 | 690 |
| focus_2022_gap | validation | 0 | 0 | 0 | 0 |
| focus_2026_may | test | 1426 | 1426 | 1472 | 1426 |
| focus_2026_may | train | 0 | 0 | 0 | 0 |
| focus_2026_may | validation | 0 | 0 | 0 | 0 |
| focus_2026_july | test | 276 | 276 | 322 | 276 |
| focus_2026_july | train | 0 | 0 | 0 | 0 |
| focus_2026_july | validation | 0 | 0 | 0 | 0 |

Counts refer to saved predictor-available training transitions for the fixed 46 grids, saved validation predictions, and saved test predictions. Split follows target date. `either_rows` is a union of both endpoints, not their sum. `target_no_activity_rows` counts constructed labels potentially requiring verification; it does not count proven wrong labels. Window counts overlap and must not be summed.

| flag | split | total_rows | origin_flagged | target_flagged | either_endpoint_flagged | flagged_targets_labeled_no_activity | affected_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_zero_records | test | 27968 | 2162 | 2162 | 2392 | 2162 | 8.552631578947368 |
| raw_zero_records | train | 105340 | 690 | 690 | 736 | 690 | 0.6986899563318777 |
| raw_zero_records | validation | 33626 | 1012 | 1012 | 1288 | 1012 | 3.830369357045144 |
| audit_suspect_union | test | 27968 | 3450 | 3450 | 3680 | 3180 | 13.157894736842104 |
| audit_suspect_union | train | 105340 | 1150 | 1150 | 1196 | 1031 | 1.1353711790393013 |
| audit_suspect_union | validation | 33626 | 1978 | 1978 | 2254 | 1850 | 6.703146374829001 |

The broader suspect union combines all raw-zero dates and the listed outage/quality envelopes, including other validation/test advisories. It is a screening upper envelope, not a measured number of invalid transitions. Partial-day boundaries are included conservatively. This audit does not modify eligibility or calculate corrected performance. Candidate transitions across all 187 grids are tabulated separately; these were not all fitted.

## Source interpretation

[NASA 2022](https://landweb.modaps.eosdis.nasa.gov/displayissue?id=477) documents unrecoverable outage loss and recommends reprocessing August 11–19; that recommendation does not prove the downloaded archive lacks reprocessing. [NASA FIRMS](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms/outages.html) distinguishes July 2026 restart on July 16 from safe use on July 27. The CSV lists calendar envelopes for other advisories; these are not exact granule masks. One March notice has an inconsistent posting/recovery chronology, so its range is explicitly exploratory. This list is not a complete historical outage inventory.

[NASA 2023](https://landweb.modaps.eosdis.nasa.gov/displayissue?id=480) independently documents the July 26–27 gap found in validation. The [FIRMS missing-data table](https://firms.modaps.eosdis.nasa.gov/api/missing_data) was checked, but its rows failed to load in the available web response; no database-wide missing-date inventory is claimed.

## What is established and what remains unresolved

- The existing 0.5° panel totals match retained raw FIRMS detections on every date; missing detections were not introduced by the spatial aggregation. The panel fills calendar cells without separate target-observation coverage checks (`scripts/prepare_modeling_panel_025.py`).
- Both source CSVs identify SNPP, so another satellite is not supplying replacements within these files. Inspect `audit_manifest.json` for product versions and dates.
- A documented full instrument outage supports unavailable observations during the outage interval. Exact valid grid-days require acquisition footprints, cloud/valid-pixel QA and archive completeness. Partial-day loss can still produce records and underestimate intensity.
- No local download receipt, granule inventory or per-grid observation mask is established by the supplied Readme, which only links to generic documentation. `data/README.md` also leaves the FIRMS download provenance fields blank. May's cause cannot be determined from these CSVs alone. Absence of confidence-retained detections is not proof of absent observation.
- A valid next-day target transition needs verified observation support on both t and t+1. Weather publication availability is a separate unresolved operational gate; chronological splitting alone cannot certify as-of forecasting.
- Do not replace every zero with missing. Preserve legitimate observed no-detection days. No target mask or exclusion is applied by this audit.

## Decision boundary

The frozen results remain reproducible descriptions of constructed labels, but target observability is not yet certified. Documentation alone cannot resolve demonstrated source gaps; source recovery and coverage evidence come before deciding whether any versioned data correction is needed. No conclusion about required refitting or corrected scores is drawn here. Existing test results cannot be treated as an unseen test for a revised experiment.

## Verification and outputs

Raw daily counts reconcile exactly with all 0.5° panel hotspot totals. Saved validation origin/destination labels match 33,626 stored transitions. Training row counts match all 138 saved state-model fit diagnostics. Frozen rows are unique, remain in the 46 fixed grids and advance one day. SHA-256 and modification times confirm 314 protected files unchanged, including raw CSVs, both panels, frozen models/configs, final reports and predictions. No inference, fitting, threshold changes, new evaluation metrics or target edits occurred.

CSV files provide daily records before/after filtering, contiguous raw-zero runs, advisory evidence, transition impacts per window/split/grid, and candidate-population counts. The manifest records input hashes and audit provenance. Reproduce with `python scripts/audit_firms_observation_coverage.py --output <new-directory>` using DuckDB, pandas, Shapely and PyArrow.
