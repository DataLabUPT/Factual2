# Factual2

**Factual2** is a Romanian fact-checking corpus of political statements labelled with
veracity verdicts, released together with **speaker-credibility history features built in
a leakage-controlled form** and ready-to-use train/validation/test splits.

Statements are fact-checks published by [factual.ro](https://www.factual.ro). The
credibility constructions follow Bucos, [*"Data Leakage in Speaker
Credibility Features: A Systematic Analysis and Correction for the LIAR2 Fake News
Benchmark"*](https://github.com/marianbucos/liar2-credibility-leakage) (IEEE Access,
2026), generalized to Factual2's five veracity classes, plus a temporal (causal)
leak-free variant enabled by per-statement review dates.

## This release does not contain the statement texts

The statement text (`content`) and the claim-time `context` description are editorial
content of factual.ro and are **not redistributed here**. Everything else is: labels,
speaker keys, dates, venues, the split assignments, the credibility count vectors, the
experiment outputs, and the full pipeline. Each row keeps its `declaration_link`, the URL
of the source verification, so the texts can be restored from your own copy of the
factual.ro extraction:

```bash
python src/rehydrate.py --source "factual-ro 20260702.csv"   # writes data/factual2.csv
python src/build_dataset.py                                   # splits, now with text
```

The split assignment, the label codes and the count vectors are deterministic given the
seed, so a rehydrated corpus reproduces the released files exactly. See
[`docs/SCHEMA.md`](docs/SCHEMA.md) for the column dictionary and the rehydration contract.

## Dataset

`data/factual2_metadata.csv` — one row per verified statement, text columns removed.

| | |
|---|---|
| Statements (valid verdicts) | **1,958** |
| Speakers | **417** |
| Review dates span | 2016-05-19 → 2026-07-01 |
| Verdicts | `false` 710 · `true` 469 · `truncated` 376 · `partially true` 317 · `partially false` 86 |
| Statements per speaker | median 2 · max 94 · 191 speakers (45.8%) appear once · 94 with ≥5 |

Statements per review year: 2016 146 · 2017 39 · 2018 73 · 2019 358 · 2020 330 · 2021 221 · 2022 94 · 2023 130 · 2024 104 · 2025 284 · 2026 179.

Speaker history is sparse, which is exactly why leak-free credibility features matter: the
honest history of a singleton speaker is all zeros, while a corpus-level (leaky) history
consists solely of that row's own label.

**Columns:** `review_date`, `author_name`, `author_link`, `affiliation`, `source_name`,
`source_link`, `label`, `declaration_link` (plus `content` and `context` after
rehydration).

`review_date` is the date the **verification was published**, not the date the statement
was made. This is deliberate: the review date is when a label becomes available, so it is
the causally correct time variable for the temporal protocol and the `lfa_temporal`
construction.

Fields that are not complete for every row:

| Field | Missing rows | Coverage |
|---|---|---|
| `source_name` | 140 | 92.8% |
| `source_link` | 140 | 92.8% |
| `affiliation` | 32 | 98.4% |

The speaker key is `author_link` (the canonical factual.ro person page), not the display
name; where the platform hosts more than one page for the same speaker, the duplicates are
folded onto one key (`segment._SPEAKER_ALIASES`).

### Label schemes

Each scheme is a self-contained dataset with an integer-coded `label`; a statement keeps
the same split across schemes.

| Scheme | Mapping (verdict → code) | Rows | Class sizes |
|---|---|---|---|
| **5c** | false=0, partially false=1, truncated=2, partially true=3, true=4 | 1,958 | 710/86/376/317/469 |
| **3c** | {false, partially false}=0, truncated=1, {partially true, true}=2 | 1,958 | 796/376/786 |
| **2p** | {false, partially false}=0, {partially true, true}=1 (drop truncated) | 1,582 | 796/786 |
| **2c** | false=0, true=1 (drop partials + truncated) | 1,179 | 710/469 |

### Splits

`data/splits/<scheme>/<protocol>/<construction>/{train,val,test}.csv` — 8:1:1 splits.

- **Protocols:** `random_seed42` (label-stratified), `temporal` (oldest→train,
  newest→test by review date), `speaker_seed42` (speaker-disjoint — no speaker appears in
  two splits, so every evaluation speaker is cold-start).
- **Constructions** (speaker-credibility count columns appended to each row):
  - `lfa` — **Leak-Free Aggregation** (the recommended features): counts from the **train
    split only**, own label excluded (leave-one-out), frozen train vectors for seen
    speakers at inference, all-zero vectors for cold-start speakers.
  - `lfa_temporal` — leak-free **and causal**: only the speaker's train statements whose
    review date is strictly earlier than the current row's.
  - `psa`, `wsa` — leaky / partially-leaky baselines, provided **only** for the ablation
    that measures the leakage; they are diagnostic, never recommended features.

Each split row adds per-class `*_counts` columns plus `total_counts`.

## Reproducing

```bash
pip install -r requirements.txt
python src/rehydrate.py --source <your factual.ro extraction>.csv
python src/build_dataset.py        # data/splits/
python src/validate_dataset.py     # release-gating integrity + leak-freeness checks
python src/ablation.py --cred-cols full,total,individual --context both --seeds 42
                                   # the released 744-run TF-IDF grid -> results/results.csv
python src/baselines.py            # majority, identity and permutation-null floors
python src/baselines.py --combined # identity vs. honest history, fused with text
python src/ablation.py --cred-repr # alternative encodings of the same histories
python src/perclass_strata.py      # per-class metrics and history-length strata
python src/model_families.py       # the PSA-LFA gap across seven classifier families
```

Every stage is pure CPU and deterministic given the split seed. Note that `ablation.py
--seeds` is the *model* seed, which the `lbfgs` solver ignores, so one value reproduces the
released numbers exactly; `results/results_splitseeds.csv` is the same grid re-run on five
regenerated partitions (`build_dataset.py --seed k` for k in 7, 13, 42, 77, 123), while the
released splits are seed 42. `results/` already contains the outputs reported in the paper,
so the tables can be checked without re-running anything.

## Layout

```
data/
  factual2_metadata.csv       base corpus, text columns removed
  splits/<scheme>/<protocol>/<construction>/{train,val,test}.csv
results/               experiment outputs reported in the paper, one file per table family
src/                   preprocess · segment · schemes · credibility · build_dataset
                       validate_dataset · ablation · baselines · perclass_strata
                       model_families · rehydrate (restores the texts)
docs/SCHEMA.md         column dictionary, results dictionary, rehydration contract
CHECKSUMS.sha256       sha256 of every released file
```

## Citation

If you use Factual2, please cite the dataset paper — Bucos & Drăgulescu, *"Factual2:
Extending a Romanian Fact-Checking Corpus with Speaker Credibility Histories for
Leak-Aware Veracity Classification"* — and the credibility-leakage methodology it builds
on (Bucos, *"Data Leakage in Speaker Credibility Features"*, IEEE Access, 2026). See
`CITATION.cff`.

```bibtex
@article{bucos2026factual2,
  author  = {Bucos, Marian and Dr\u{a}gulescu, Bogdan},
  title   = {Factual2: Extending a Romanian Fact-Checking Corpus with Speaker
             Credibility Histories for Leak-Aware Veracity Classification},
  journal = {Electronics},
  year    = {2026},
  note    = {Manuscript in preparation}
}

@article{bucos_liar2_credibility_leakage,
  author  = {Bucos, Marian},
  title   = {Data Leakage in Speaker Credibility Features: A Systematic
             Analysis and Correction for the LIAR2 Fake News Benchmark},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3730178}
}
```

## License

Everything in this repository — the code, the split assignments, the label codings, the
credibility count vectors, the results and the documentation — is released under the
**Apache License 2.0** (see `LICENSE`).

The release carries no statement text: `content` and `context` are not reproduced. What is
released alongside the derived artifacts is factual record — the verdict, the review date,
the speaker, the affiliation, the venue and the URLs — and every row keeps the
`declaration_link` of its source verification on [factual.ro](https://www.factual.ro), both
as attribution and so that any label can be checked at the source.
