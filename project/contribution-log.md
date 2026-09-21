# Contribution Log

This repository's commit history **is** the verified contribution record. Each member commits under their own name — a commit attributed to someone who did not do the work invalidates the log.

---

## CRediT role assignment

| CRediT role | Member |
|---|---|
| Conceptualization | Member 1 (Lead) |
| Methodology | Member 1 |
| Formal Analysis | Member 1 |
| Project Administration | Member 1 |
| Writing — Original Draft | Member 1 |
| Data Curation | Member 2 (Data/Software) |
| Software | Member 2 |
| Validation | Member 2, Member 3 |
| Visualization | Member 2 |
| Literature / Investigation | Member 3 (Literature/Review) |
| Analysis | Member 3 |
| Writing — Review & Editing | Member 3 |

> Replace "Member 1/2/3" with GitHub handles and full names before the midterm. The mapping must match what appears in the manuscript's author contribution statement.

---

## Week-by-week log

| Week | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| 1 — Project setup | CRediT plan, repo, data/AI/authorship rules | repo scaffolding | — |
| 2 — Issue brief + dataset audit | issue brief (7 components), A1–A5 alternatives | dataset audit (fire/smoke dataset, later discontinued) | stakeholder & criteria review |
| 3 — Journal-fit | backward schedule, project charter | GEE account, FIRMS VIIRS Sumatra download started, 10 km grid | journal candidate verification, literature collection began |
| 4 — Literature sprint | search protocol, state of the art, gap synthesis | metadata retrieval & verification, matrix build | collection of 7 key PDFs, screening, review |

### Week 4 detail

| Member | Contribution | Evidence in repo |
|---|---|---|
| Member 3 | Collected the 7 key papers (`C5` cluster), screened candidates | `literature/metadata/team_collected_works_metadata.json` |
| Member 2 | Retrieved and verified metadata for 256 works via OpenAlex/Crossref; built the matrix | `literature/matriks-literatur.csv`, `literature/metadata/works_metadata.json` |
| Member 1 | Wrote search protocol, state of the art, gap synthesis, novelty statement | `docs/search-protocol.md`, `literature/Minggu4_Literature_Sprint.md` |

---

## Rules in force

1. **Each member commits under their own name.** Do not batch a teammate's work into your own commit.
2. **Every dataset requires a provenance entry** before it enters the pipeline — see `docs/data-provenance.md`.
3. **Dual submission is prohibited.**
4. **Generative AI** may be used only as an aid. Its use must be disclosed, and its output must be verified by a human. Unverified AI output must not enter the manuscript.
5. **No number without a source.** Period and institution must be stated alongside any figure.
6. A contribution is only logged once the artifact is committed.

---

## Open items

- [ ] Map Member 1/2/3 to real names and GitHub handles
- [ ] Each member makes their first commit under their own account
- [ ] Verify `JQMA 2025` authorship (missing in OpenAlex and Crossref)
- [ ] Fill the dataset provenance table during the data sprint
- [ ] Verify target journal SINTA rank and APC (check `sinta.kemdiktisaintek.go.id`)
