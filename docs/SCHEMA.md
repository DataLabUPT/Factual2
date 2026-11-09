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

## Results files: `results/`

Each table of the paper is read from one of these files. All are plain CSV with a header
row, all are regenerable with the commands in the README, and none of them contains
statement text.

| File | Produced by | Contents |
|---|---|---|
| `results.csv` | `ablation.py` | The reference grid: one row per scheme x protocol x construction x feature group x `cred_cols` x `context`, with validation and test accuracy and macro-F1. |
| `results_splitseeds.csv` | `build_dataset.py --seed k` then `ablation.py` | The same grid re-run on five regenerated partitions (k = 7, 13, 42, 77, 123) and aggregated, which is where the error bars come from: the model seed is inert for this pipeline, only the split seed moves the numbers. |
| `baselines.csv` | `baselines.py` | Per scheme x protocol: the majority-class floor, the speaker-identity model, the zero-feature control, and a label-permutation null over 200 draws (mean, SD and 95th percentile). |
| `baselines_combined.csv` | `baselines.py --combined` | Seven feature groups per split seed, from `identity` and `lfa` alone to `text+identity+lfa`, which decomposes how much the honest history adds on top of speaker identity. |
| `cred_repr.csv` | `ablation.py --cred-repr` | The same histories re-encoded as raw counts, class proportions, and proportions or counts plus log(1 + total), with the feature count of each encoding. |
| `perclass.csv` | `perclass_strata.py` | Precision, recall, F1 and test support for every class of a scheme, per split seed, so the variability of the rare verdicts stays visible. |
| `strata.csv` | `perclass_strata.py` | The reference classifier scored on test rows grouped by admissible history length (empty, 1-4, >= 5 past verdicts), next to the train-majority predictor on the same rows. |
| `model_families.csv` | `model_families.py` | The counts-only PSA-versus-LFA comparison repeated with seven classifiers of differing inductive bias, per scheme x protocol x split seed, with the per-run gap. |

The `context` column of `results.csv` is an ablation axis (the platform's claim-time context
folded into the text features, on or off), not the context text itself.

Transformer results are not part of this release: the encoder arm is not shipped, so its
output file is not either. Everything above runs on CPU.

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
