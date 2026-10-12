# Decision Making — Retrospective Hotspot State Transitions (ITERA)

> **Current status — 11 October 2026:** Active scope is retrospective detected-state transition research (DEC-12), without patrol allocation. REV-01 input reconstruction, REV-02 quality audit and conservative REV-03 correction/refit are verified. The revised analysis retains 46 grids and 138 models on 23,690 test transitions; it is post-hoc and conditional on provisional detected-export usability, not validated observation coverage or operational forecasting. Read the [current results and methods](docs/FINAL_RESULTS.md), [progress](docs/PROGRESS.md), [audit](docs/FINAL_AUDIT.md), [plan](docs/REVISION_PLAN.md), [context](docs/PROJECT_CONTEXT.md), and [decisions](docs/DECISIONS.md). Full manuscript/slides remain unreviewed; historical figures below are not revised results. Heavy inputs/models are local and ignored by Git. Local HEAD remains `66bf5a8`, cached final commit `0745186`; no merge/push is implied. [AGENTS.md](AGENTS.md) defines maintenance instructions.

## Historical local proposal — superseded by DEC-12

The original proposal/status text below is retained as dated evidence, including obsolete patrol goals, missing-source statements and old results. It is not the current project specification. The exact pre-REV-05 README bytes are archived in `docs/attachments/rev05_20261011/README_before.txt`.

Research project for **Decision Making (SD25-41302)**, Data Science, Institut Teknologi Sumatera (ITERA).

**Working title:** *Prioritisasi Wilayah Patroli Kebakaran Hutan dan Lahan di Sumatra Menggunakan Covariate-Dependent Markov Chain*

**Decision problem:** which spatial grid cells should be prioritized for forest-and-land-fire (karhutla) patrol on the **next day**, given limited patrol capacity.

**Decision maker:** fire monitoring / patrol agencies (Manggala Agni–KLHK, BPBD).

---

## Status

### Implementation and external results — checked 8 October, documented 9 October 2026

The local implementation uses a **0.1° grid and binary hotspot states**. Preprocessing files and completion logs exist in the current workspace; large data and outputs remain outside Git. The annual batch folder is empty, `src.models` is absent, and all four notebooks have no saved execution counts or outputs. Their analyses, memory estimates, training results, and research-question conclusions are **draft expectations, not demonstrated results**. The locally present `batch_builder.py` still needs review; see the [audit findings](docs/audit-final-artifacts.md).

A group member supplied a separate final-reporting ZIP using a **0.5° grid, three FRP-based states, 46 eligible grids, and 138 independent multinomial transition models**. Its summary reports 27,968 test observations and 86.09% accuracy. The [preserved reporting files](docs/attachments/final_artifacts/final_artifacts_summary.md) and [audit](docs/audit-final-artifacts.md) show consistent tables/figures, but all eight provenance-listed source files are missing locally. Training, eligibility, leakage controls, AUC/calibration, and reproducibility remain unverified. Operational patrol Recall@K/Precision@K results are not supplied.

The always-No-Activity baseline has **85.99% accuracy** on the reported support, so the reported 86.09% accuracy needs to be read alongside **43.22% balanced accuracy** and class-specific recall. These are externally reported results and arithmetic comparisons, not results produced by the local notebooks. The source package needed for further verification is listed in the [request checklist](docs/final-artifacts-source-request.md).

The research title and patrol-ranking goal above remain the working proposal; adopting the external design requires a separate methodological reconciliation. Dataset provenance remains incomplete. The historical schedule below records the original course plan and is not an updated implementation-progress claim.

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
│   ├── data-provenance.md                # sources and verification status
│   ├── audit-final-artifacts.md          # local/external design comparison and evidence
│   ├── final-artifacts-source-request.md # missing-source checklist for the group
│   └── attachments/final_artifacts/      # unchanged external reporting files + audit metadata
├── notebooks/                           # four draft workflow notebooks
├── src/                                 # local 0.1° preprocessing implementation
├── scripts/                             # preprocessing runner and read-only artifact verifier
└── data/README.md                        # data access and provenance template; data outside Git
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

Literature reproducibility: every entry in the matrix carries a **resolvable DOI**, as documented during the literature sprint. Metadata was retrieved programmatically from OpenAlex and Crossref; nothing was written from memory. The exact queries are in [`docs/search-protocol.md`](docs/search-protocol.md), and per-number sourcing is in [`docs/data-provenance.md`](docs/data-provenance.md). This literature record does not establish model reproducibility; the external experiment still requires its original source package.

Regenerate/verify metadata:

```bash
# resolve any DOI in the matrix
curl -s -o /dev/null -w '%{http_code}\n' "https://api.openalex.org/works/https://doi.org/<DOI>"

# look up a work by title
curl -s "https://api.openalex.org/works?search=<query>&per-page=5&select=doi,title,publication_year,cited_by_count"
```

---

## Team & contribution rules

Three members, roles assigned under **CRediT**. Contribution tracking is maintained separately; the former `project/` folder was removed from the repository.

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
