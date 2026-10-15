# Factual2 — schema, provenance, and rehydration

## Base corpus: `data/factual2_metadata.csv`

1,958 rows, one per published factual.ro verification carrying one of the five veracity
verdicts. Procedural verdicts (*impossible to check*, *outside mandate*, *only with
institutional support*, *in mandate*, *verification in progress*) are excluded upstream.

| Column | Type | Description |
|---|---|---|
| `review_date` | `dd.mm.yyyy` | Date the **verification** was published (not the statement date). Time key for the temporal protocol and for `lfa_temporal`. |
| `author_name` | text | Display name of the speaker, as shown by the platform. |
| `author_link` | URL | Canonical person page — **the speaker key**. Duplicate pages for one speaker are folded onto a single key (`segment._SPEAKER_ALIASES`). |
| `affiliation` | text | Speaker affiliation at the time of the statement, where recorded. |
| `source_name` | text | Venue where the claim was made (e.g. Facebook, Digi24). |
| `source_link` | URL | Link to that venue item, where recorded. |
| `label` | enum | Verdict: `false`, `partially false`, `truncated`, `partially true`, `true` (English vocabulary mapped from the platform's Romanian verdicts). |
| `declaration_link` | URL | The source verification. Unique per row; the join key for rehydration and the per-row attribution. |
| `content` | text | **Not released.** The statement. Restored by `src/rehydrate.py`. |
| `context` | text | **Not released.** Claim-time context written by the platform. Restored by `src/rehydrate.py`. |

Two editorial fields of the source verification — the framing of what is checked and the
conclusion — are deliberately absent from Factual2 altogether: both are written *after* the
verdict and paraphrase it, so a model reading them would be reading the label.

## Split files: `data/splits/<scheme>/<protocol>/<construction>/{train,val,test}.csv`

Each file carries the base columns above (minus the text), the scheme's integer `label`,
then the credibility columns for that construction: one `*_counts` column per class of the
scheme, plus `total_counts`.

Segmentation runs **once on the base corpus**, so a statement keeps the same split across
all schemes and constructions — this is what makes constructions comparable. `temporal` is
deterministic and carries no seed suffix; `random` and `speaker` carry `_seed<k>`.

## Credibility constructions

For a statement `s` by speaker `p`, the count vector holds, per class of the active scheme,
how many of `p`'s statements fall in that class under the construction's aggregation rule.

| Construction | Rule | Status |
|---|---|---|
| `lfa` | Train split only; training rows leave-one-out; evaluation rows of a train-seen speaker get the frozen train vector; unseen speaker → zero vector. | **released, default** |
| `lfa_temporal` | As `lfa`, further restricted to the speaker's training statements with a **strictly earlier** review date. | **released, causal** |
| `psa` | Whole-corpus counts, **including the row's own label**. | leaky — diagnostic only |
| `wsa` | Within-split counts, own label excluded; evaluation labels still circulate inside the evaluation split. | partially leaky — diagnostic only |

`src/validate_dataset.py` re-derives all four constructions independently (from split
assignments, labels and dates only — it does not import `credibility.py`) over all 48
scheme × protocol × construction combinations and asserts LFA leak-freeness. It exits
non-zero on failure and gates the release.

## Rehydration contract

`src/rehydrate.py --source <extraction>.csv` joins on `declaration_link` and writes
`data/factual2.csv`. The source may be the raw factual.ro extraction (`declaration link`,
`content`, `context`) or any CSV with those columns; space-separated headers are accepted.
Coverage is reported per file and the script aborts if any row is left without text unless
`--allow-partial` is given.

After rehydration, `python src/build_dataset.py` regenerates every split file with text.
Because segmentation and the count constructions are deterministic given the seed, the
regenerated files match the released ones on every non-text column; `CHECKSUMS.sha256`
covers the released (text-free) files as shipped.
