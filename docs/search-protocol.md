# Search Protocol — Week 4 Literature Sprint

Reproducible search protocol for the karhutla patrol-prioritization literature review.

**Databases:** OpenAlex API (primary, programmatic, metadata-complete) · Crossref API (DOI resolution, missing-author fallback) · semantic web search for cross-domain method discovery.

**Retrieved:** 256 unique works → 117 passed title relevance → **42** entered the matrix (35 from the systematic pass, 7 collected manually by the team).

---

## Query strings

Two passes were run for every cluster. Pass A (`search`, relevance-ranked) gives breadth; pass B (`title_and_abstract.search`, citation-ranked) gives precision. **The union of both passes was used** — neither alone surfaced the domain papers.

### Cluster 1 — Prediction & susceptibility

| Pass | Query string |
|---|---|
| A | `hotspot prediction forest fire machine learning` |
| A | `forest fire susceptibility mapping machine learning` |
| A | `VIIRS MODIS active fire hotspot validation Indonesia` |
| A | `drought peatland fire risk El Nino Indonesia` |
| B | `hotspot forest fire Indonesia machine learning prediction` |
| B | `karhutla peatland fire Indonesia risk` |

### Cluster 2 — Remote sensing & data quality

| Pass | Query string |
|---|---|
| A | `VIIRS MODIS active fire hotspot validation Indonesia` |
| A | `global biomass burning emission datasets` |
| B | `operational fire danger forecast system daily prediction` |
| B | `FIRMS hotspot confidence accuracy Indonesia fire` |

### Cluster 3 — Markov & spatio-temporal modelling

| Pass | Query string |
|---|---|
| A | `non-homogeneous Markov chain covariate dependent transition` |
| A | `Markov chain fire risk transition probability` |
| A | `Markov chain cellular automata land cover change prediction` |
| A | `hidden Markov model wildfire state transition` |
| B | `multistate Markov model covariate dependent transition probabilities` |
| B | `Markov chain forest fire occurrence probability daily` |
| B | `Markov chain weather model fire danger rating` |

### Cluster 4 — Decision, patrol & resource allocation

| Pass | Query string |
|---|---|
| A | `fire patrol resource allocation prioritization optimization` |
| A | `fire suppression resource allocation optimization wildfire` |
| A | `decision support system hotspot ranking early warning` |
| A | `top-K evaluation ranking resource allocation emergency` |
| B | `Manggala Agni patrol forest fire Indonesia` |
| B | `pencegahan kebakaran hutan lahan patroli Indonesia hotspot` |
| B | `priority area determination forest fire prevention` |

---

## API request templates

```bash
# Pass A — breadth, relevance ranked
curl -s "https://api.openalex.org/works?search=<QUERY>&filter=publication_year:>2014&per-page=12&sort=relevance_score:desc&select=id,doi,title,publication_year,cited_by_count,primary_location,authorships,type"

# Pass B — precision, title+abstract scoped, citation ranked
curl -s "https://api.openalex.org/works?filter=title_and_abstract.search:<QUERY>,publication_year:>2009&per-page=10&sort=cited_by_count:desc&select=id,doi,title,publication_year,cited_by_count,primary_location,authorships,type"

# Bulk abstract retrieval for shortlisted works
curl -s "https://api.openalex.org/works?filter=ids.openalex:W1|W2|W3&per-page=40&select=id,doi,title,publication_year,cited_by_count,primary_location,authorships,abstract_inverted_index"

# Crossref fallback (missing authors / citation counts)
curl -s "https://api.crossref.org/works/<DOI>"
curl -s "https://api.crossref.org/works?query.bibliographic=<TITLE>&rows=2"
```

### Retrieval gotchas encountered

- **Always pass `select=`.** Full records are megabytes per query and truncate mid-JSON.
- **Write responses to file and parse the file.** Some `title_and_abstract.search` responses contain raw control characters that break in-memory parsing.
- **A zero result usually means the filter, not the query.** `title_and_abstract.search` requires every term; a four-word query can return 0 while `search` returns thousands. Drop the least essential word if a query unexpectedly returns nothing.
- **OpenAlex occasionally has empty `authorships`.** Fall back to Crossref; if Crossref also lacks authors, label the row by journal+year and flag it rather than inventing a name.
- **Preprints duplicate the version of record.** Keep the journal version and check the year follows it.

---

## Inclusion criteria

1. Published 2005–2026 (earlier only for irreplaceable canonical references).
2. Relevant to at least one of: fire prediction/susceptibility, satellite active-fire products, Markov state/transition modelling, or resource allocation and decision prioritization.
3. Sufficient method/abstract information to extract into the matrix.
4. Priority given to: (i) Indonesia / Southeast Asia context, (ii) peatland, (iii) Sumatra (Riau, Jambi, South Sumatra).

## Exclusion criteria

1. Structural/building fire modelling (closed-system fire).
2. Marine/vessel fire.
3. Pure fire-spread simulation and suppression logistics with no prioritization component.
4. Landslide / erosion / desertification studies that merely happen to use Markov or ML.

## Manual collection (Cluster C5)

Seven papers were collected directly by the team from publisher sites and verified against Crossref + OpenAlex:

`Komara 2016` · `Widodo 2014` · `Afrizal 2025` · `Singh 2021` · `Sakti 2022` · `Zakaria 2019` · `Tan 2020`

These are tagged `C5` in the matrix so systematic-search results and manual findings remain distinguishable for provenance purposes.
