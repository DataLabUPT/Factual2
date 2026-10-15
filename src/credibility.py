"""Speaker-credibility count-vector constructions for Factual2.

Implements the count-vector definitions from Bucos, "Data Leakage in Speaker
Credibility Features: A Systematic Analysis and Correction for the LIAR2 Fake
News Benchmark" (IEEE Access, 2026), plus a temporal leak-free variant that
exploits Factual2's per-statement review dates (`review_date` = when the verdict
was published, which is when a label becomes available to count). The functions
are label-space agnostic: they
work over whatever class set the active label scheme defines (see schemes.py).

For a statement by speaker p in class c, the count vector holds, per class, how
many of p's statements fall in that class. The constructions differ in
aggregation scope and self-exclusion of the row's own label:

  PSA  Pre-Split Aggregation (leaky baseline): whole-corpus counts INCLUDING the
       row's own label; identical vector for every row of a speaker.
  WSA  Within-Split Aggregation: counts within the row's own split, own label
       excluded (floor 0). Removes self-inclusion but keeps within-split leakage.
  LFA  Leak-Free Aggregation (valid): counts from the TRAIN split only. A train
       row excludes its own label (floor 0); a val/test row of a train-seen
       speaker gets that speaker's full train vector; a cold-start speaker gets
       an all-zero vector.
  LFA_TEMPORAL (Factual2 extension): leak-free AND causal. Counts come from the
       speaker's TRAIN statements whose review date is strictly earlier than the
       current row's (a verdict counts only once it has been published).

`total_counts` is the sum of the class counts.
"""
from collections import defaultdict

from segment import _speaker_key, _parse_date


def _empty(classes):
    return {c: 0 for c in classes}


def _tally(rows, classes):
    t = defaultdict(lambda: _empty(classes))
    for r in rows:
        t[_speaker_key(r)][r["label"]] += 1
    return t


def _to_cols(vec, classes, count_names):
    out = {count_names[c]: vec[c] for c in classes}
    out["total_counts"] = sum(vec[c] for c in classes)
    return out


def _psa(rows, splits, classes):
    glob = _tally(rows, classes)
    return {r["declaration_link"]: dict(glob[_speaker_key(r)]) for r in rows}


def _wsa(rows, splits, classes):
    per_split = {s: _tally([r for r in rows if splits[r["declaration_link"]] == s], classes)
                 for s in ("train", "val", "test")}
    result = {}
    for r in rows:
        vec = dict(per_split[splits[r["declaration_link"]]][_speaker_key(r)])
        vec[r["label"]] = max(0, vec[r["label"]] - 1)
        result[r["declaration_link"]] = vec
    return result


def _lfa(rows, splits, classes):
    train = _tally([r for r in rows if splits[r["declaration_link"]] == "train"], classes)
    result = {}
    for r in rows:
        sp = _speaker_key(r)
        if splits[r["declaration_link"]] == "train":
            vec = dict(train[sp])
            vec[r["label"]] = max(0, vec[r["label"]] - 1)   # leave-one-out in train
        elif sp in train:
            vec = dict(train[sp])                            # frozen train vector
        else:
            vec = _empty(classes)                            # cold-start
        result[r["declaration_link"]] = vec
    return result


def _lfa_temporal(rows, splits, classes):
    train_by_speaker = defaultdict(list)
    for r in rows:
        if splits[r["declaration_link"]] == "train":
            train_by_speaker[_speaker_key(r)].append(r)
    for sp in train_by_speaker:
        train_by_speaker[sp].sort(key=lambda r: _parse_date(r["review_date"]) or (0, 0, 0))

    result = {}
    for r in rows:
        sp = _speaker_key(r)
        d = _parse_date(r["review_date"])
        vec = _empty(classes)
        if d is not None:
            for h in train_by_speaker.get(sp, []):
                hd = _parse_date(h["review_date"])
                if hd is None:
                    continue
                if hd < d:                    # strictly earlier -> excludes self & future
                    vec[h["label"]] += 1
                else:
                    break                     # sorted; nothing later qualifies
        result[r["declaration_link"]] = vec
    return result


CONSTRUCTIONS = {
    "psa": _psa,
    "wsa": _wsa,
    "lfa": _lfa,
    "lfa_temporal": _lfa_temporal,
}


def credibility_columns(rows, splits, classes, count_names, construction="lfa"):
    """Returns {declaration_link -> count-column dict} for the chosen construction,
    over the given class set / column naming."""
    vectors = CONSTRUCTIONS[construction](rows, splits, classes)
    return {link: _to_cols(vec, classes, count_names) for link, vec in vectors.items()}
