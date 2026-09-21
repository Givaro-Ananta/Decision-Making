# Data Provenance

Rule in force: **no number enters any document without a traceable source, a stated period, and a resolvable identifier.** This file records where each figure came from.

Last updated: Week 4 literature sprint.

---

## 1. Literature metadata

| Field | Source | Method |
|---|---|---|
| Title, year, venue, DOI, citation count, authorship | **OpenAlex API** (`api.openalex.org/works`) | programmatic retrieval, one request per query batch |
| Abstracts (screening only, not redistributed) | OpenAlex `abstract_inverted_index` | reconstructed by sorting word positions |
| Missing author lists | **Crossref API** (`api.crossref.org/works/<DOI>`) | fallback lookup |
| Citation count for works absent from OpenAlex | Crossref `is-referenced-by-count` | `Afrizal 2025` only |

**Verification status:** all **42** matrix entries carry a resolvable DOI (HTTP 200 from `api.openalex.org/works/<doi>`). No entry was written from memory.

One entry is flagged: `JQMA 2025` has **no author list** in either OpenAlex or Crossref. It is labelled by journal + year, and must be verified manually on the journal page before being cited in the manuscript.

---

## 2. Urgency figures (karhutla magnitude)

These are the citable numbers available for the introduction. All come from **peer-reviewed or institutional publications** — deliberately preferred over government portals, because these carry permanent identifiers and a stated period.

| Figure | Value as published | Period | Source | DOI / identifier |
|---|---|---|---|---|
| Riau burnt area | 5,434 ha | up to 29 Mar 2014 | `Komara 2016` citing Dinas Kehutanan Riau & Manggala Agni | 10.28932/jutisi.v2i3.524 |
| Riau hotspot count | 1,272 titik | to 29 Mar 2014 | same as above | same |
| Jambi hotspots | 1,678 titik | 2003 | `Widodo 2014` citing Kementerian Kehutanan 2006 | 10.14710/pwk.v10i2.7643 |
| Jambi burnt area | 3,025 ha | 2003 | same | same |
| Jambi hotspots | 6,948 titik | 2006 | same | same |
| Jambi burnt area | 2,408.10 ha (1,227.60 in forest area + 1,180.50 outside) | 2006 | same | same |
| Indragiri Hulu burnt area | >1,600 ha | last 5 years before 2025 | `Afrizal 2025` | 10.33084/daun.v12i1.9750 |
| Indonesia high-priority fire area | 379,516 km² | published 2022 | `Sakti 2022` | 10.3390/rs14030543 |
| Priority areas since burnt | 19.50% of priority areas experienced wildfire-caused deforestation in the prior 10 years | published 2022 | `Sakti 2022` | 10.3390/rs14030543 |
| 1997/1998 fires, Sumatra | 11.69 million ha total Indonesia; 2.07 million ha Sumatra | 1997/1998 | cited in `Komara 2016` and `Widodo 2014` (secondary) | see above |

### Measures are NOT interchangeable

- **Hotspot count** = satellite thermal anomalies (may include false detections, multiple per fire).
- **Burnt area (hektare)** = mapped/estimated area, different instrument and definition.
- **Deforestation** ≠ fire; `Singh 2021` measures forest loss, of which fire is one driver.

Do not mix these in a single sentence without saying which measure is which.

---

## 3. Not yet verified — do not cite

| Item | Status | Action required |
|---|---|---|
| SiPongi / KLHK hotspot statistics for Sumatra by province | **not verified** — automated retrieval did not complete | retrieve manually at `sipongi.menlhk.go.id`; record URL + period + screenshot |
| BNPB DIBI burnt-area figures | **not verified** | retrieve manually at `dibi.bnpb.go.id` |
| Confidence-weighted VIIRS hotspot count | derived later from the FIRMS dataset during the data sprint | document the extraction script and date |
| Target journal SINTA rank and APC | **not verified** — SINTA accreditation has a validity period and expires | check `sinta.kemdiktisaintek.go.id` in a real browser; screenshot the APC page |

---

## 4. Dataset provenance (to be completed during the data sprint)

| Dataset | Provider | Variables used | Extraction method | Date pulled |
|---|---|---|---|---|
| NASA FIRMS VIIRS active fire | NASA FIRMS | hotspot, FRP, confidence | to be documented (GEE / FIRMS API) | — |
| ERA5-Land | Copernicus CDS | temperature, wind | to be documented | — |
| CHIRPS | UCSB CHC | daily rainfall + 3/7-day rolling | to be documented | — |
| ESA WorldCover | ESA | land cover class | to be documented | — |
| Peatland extent (optional) | BRGM / KLHK | peat depth / extent | to be documented | — |

**Rule:** each row must be filled with the actual extraction script and pull date before the dataset enters modelling. A dataset without provenance does not enter the pipeline.

---

## 5. Verified external findings that shaped the method

| Finding | Source | Consequence for this project |
|---|---|---|
| Significant covariates for Indonesian fires **change** between 30% and 80% hotspot detection confidence thresholds | `Tan 2020`, Int. J. Wildland Fire, 10.1071/wf20036 | detection-threshold sensitivity is promoted from optional to **part of RQ3** |
| Markov chain applied to a karhutla-related air pollution index, with a statement that the model can support government prevention planning | `Zakaria 2019`, Sustainability, 10.3390/su11195190 | "first to use Markov for fire" is **retracted** as a novelty claim; only covariate-stratified, per-grid, t+1 version remains claimable |
| Spatial prioritization model for Indonesian wildfire mitigation; authors affiliated with ITB and ITERA | `Sakti 2022`, Remote Sensing, 10.3390/rs14030543 | used as **foundation and context**, not as a claim of novelty; must be cited prominently |
| Hotspot ranking decision-support for Riau including a "distance from patrol post" criterion | `Komara 2016`, JUTISI, 10.28932/jutisi.v2i3.524 | provides the **framework for defining K** as post-based capacity |
| "Patroli hotspot dan pengendalian api tidak dapat direncanakan dengan lokasi dan tata waktu yang efektif" without a risk map | `Widodo 2014`, J. Pembangunan Wilayah & Kota, 10.14710/pwk.v10i2.7643 | local (Jambi) justification for the operational need |
| Identifying emerging hotspots can direct **limited resources** | `Singh 2021`, Ecology and Evolution, 10.1002/ece3.7562 | supports the limited-capacity framing and hotspot typology |
