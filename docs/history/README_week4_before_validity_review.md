# Decision Making — Karhutla Patrol Prioritization (ITERA)

Research project for **Decision Making (SD25-41302)**, Data Science, Institut Teknologi Sumatera (ITERA).

**Working title:** *Prioritisasi Wilayah Patroli Kebakaran Hutan dan Lahan di Sumatra Menggunakan Covariate-Dependent Markov Chain*

**Decision problem:** which spatial grid cells should be prioritized for forest-and-land-fire (karhutla) patrol on the **next day**, given limited patrol capacity.

**Decision maker:** fire monitoring / patrol agencies (Manggala Agni–KLHK, BPBD).

---

## Status

| Week | Checkpoint | Status |
|---|---|---|
| 1 | Project setup (CRediT, repo, rules) | done |
| 2 | Issue brief + dataset audit | done — approved by supervisor |
| 3 | Journal-fit (target journal locked) | in progress |
| **4** | **Literature sprint (matrix + gap)** | **documented in this repo** |
| 5 | Research positioning (freeze title, RQs, novelty) | next |
| 6 | Method protocol | planned |
| 7 | Proposal assembly + pipeline smoke test | planned |
| 8 | Midterm — proposal freeze | planned |
| 9–13 | Data sprint, cleaning, modelling, evaluation | planned |
| 14 | Manuscript sprint (IMRaD) | planned |
| 15 | Internal editorial board, revision, cover letter | planned |
| 16 | Final — submission day (OJS evidence) | planned |

---

## Repository layout

```
.
├── README.md
├── literature/
│   ├── Minggu4_Literature_Sprint.md      # search protocol, matrix, state of the art, gaps
│   ├── matriks-literatur.csv             # 42 entries x 16 columns (open in Excel/Sheets)
│   └── metadata/
│       ├── works_metadata.json           # slim, citation-safe metadata for all 42 works
│       └── team_collected_works_metadata.json
├── docs/
│   ├── search-protocol.md                # reproducible query strings + inclusion/exclusion
│   └── data-provenance.md                # where every number came from
└── project/
    └── contribution-log.md               # CRediT roles + per-member log (see rules below)
```

---

## Week 4 literature sprint — result at a glance

- **256** works retrieved from OpenAlex + **7** papers collected manually by the team
- **117** passed title relevance screening
- **42** entered the matrix — C1 prediction (12), C2 remote sensing (6), C3 Markov/spatio-temporal (8), C4 decision/patrol (9), **C5 team-collected (7)**

Clusters:

| ID | Cluster | What it covers |
|---|---|---|
| C1 | Prediction & susceptibility | ML hotspot prediction, susceptibility mapping, imbalance handling |
| C2 | Remote sensing & data | active fire products, burned area, fire weather indices |
| C3 | Markov & spatio-temporal | CA-Markov, covariate-dependent transition frameworks, HMM |
| C4 | Decision, patrol, allocation | resource allocation, optimization, patrol systems |
| C5 | Team-collected | manually sourced PDFs, verified against Crossref + OpenAlex |

### Three research gaps (tiered)

1. **Model level** — no hotspot state-transition model that is **conditioned on covariates**, per grid, at daily t+1 horizon. (`Zakaria 2019` used Markov on a karhutla-related air pollution index, but with a **homogeneous** transition matrix on a **single location** and steady-state output.)
2. **Decision level** — existing prioritization (`Komara 2016`; `Sakti 2022`) operates on **already-detected** hotspots or static susceptibility; none prioritizes **next-day occurrence probability** and evaluates it with Recall@K at a capacity-meaningful K.
3. **Data level** — `Tan 2020` showed that significant covariates for Indonesian fires **change** when the hotspot detection confidence threshold moves from 30% to 80%; no prioritization study tests the stability of its ranking against this.

### Novelty statement

> This work does not claim to be first at Markov chains or at hotspot prioritization. It fills the remaining intersection: **covariate-stratified hotspot state-transition probabilities on a daily grid, evaluated as a limited-capacity prevention ranking (Recall@K), with a sensitivity test against satellite detection confidence thresholds.**

---

## Reproducibility

Every entry in the matrix carries a **resolvable DOI**. Metadata was retrieved programmatically from OpenAlex and Crossref; nothing was written from memory. The exact queries are in [`docs/search-protocol.md`](docs/search-protocol.md), and per-number sourcing is in [`docs/data-provenance.md`](docs/data-provenance.md).

Regenerate/verify metadata:

```bash
# resolve any DOI in the matrix
curl -s -o /dev/null -w '%{http_code}\n' "https://api.openalex.org/works/https://doi.org/<DOI>"

# look up a work by title
curl -s "https://api.openalex.org/works?search=<query>&per-page=5&select=doi,title,publication_year,cited_by_count"
```

---

## Team & contribution rules

Three members, roles assigned under **CRediT**. See [`project/contribution-log.md`](project/contribution-log.md).

**Rules in force for this project:**

- Each member commits **under their own name** — the commit history is the contribution record.
- Every dataset used must have a documented **provenance** entry before it enters the pipeline.
- **Dual submission is prohibited.**
- Generative AI is permitted only as a tool: its use must be **disclosed**, and all output must be **verified** by a human before entering the manuscript.
- No number enters any document without a traceable source (see `docs/data-provenance.md`).

---

## Notes

- Copyrighted publisher PDFs are **not** included in this repository. Only metadata and DOIs are stored; approved full text is kept in the shared team drive.
- Presentation scripts and anticipated-question notes are maintained separately and intentionally **not** committed here.
