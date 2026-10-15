"""Train/validation/test segmentation for Factual2.

Produces an 8:1:1 split (matching the LIAR2 credibility-leakage study) under three
protocols. The split assignment is what the leak-free credibility constructions
(see credibility.py) are computed against, so segmentation must happen first.

Protocols:
  - random   : label-stratified random split (default; paper-aligned).
  - temporal : oldest 80% -> train, next 10% -> val, newest 10% -> test, ordered
               by `review_date` (verdict publication date) — deliberately: the
               review date governs when a label becomes available for training.
  - speaker  : speaker-disjoint split (no speaker appears in two splits); under
               LFA this makes every val/test speaker cold-start by construction.

Each function returns a dict {declaration_link -> 'train'|'val'|'test'}.
"""
import random
from collections import defaultdict

# The five English veracity classes of the base corpus (data/factual2.csv).
CLASSES = ["false", "true", "partially true", "partially false", "truncated"]

RATIOS = (0.8, 0.1, 0.1)  # train, val, test


# factual.ro hosts more than one person page for a few speakers, so the raw
# author_link would count them as distinct speakers and let one speaker's
# statements straddle two splits under the speaker-disjoint protocol. Each
# duplicate page is folded into the page carrying the majority of the rows.
_SPEAKER_ALIASES = {
    "https://www.factual.ro/persoane/aur-alianta-pentru-unirea-romanilor/":
        "https://www.factual.ro/persoane/aur/",
    "https://www.factual.ro/persoane/serban-nicolae-2/":
        "https://www.factual.ro/persoane/serban-nicolae/",
}


def _speaker_key(row):
    key = (row.get("author_link") or "").strip() or (row.get("author_name") or "").strip()
    return _SPEAKER_ALIASES.get(key, key)


def _parse_date(s):
    """factual.ro dates are dd.mm.yyyy; returns a sortable (y, m, d) or None."""
    parts = (s or "").strip().split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        d, m, y = parts
        return (int(y), int(m), int(d))
    return None


def _assign_counts(n, ratios=RATIOS):
    """Number of items going to train/val/test for a group of size n."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def split_random(rows, seed=42, ratios=RATIOS):
    """Label-stratified random 8:1:1 split."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    assign = {}
    for label, group in by_label.items():
        idx = list(range(len(group)))
        rng.shuffle(idx)
        n_train, n_val, _ = _assign_counts(len(group), ratios)
        for rank, i in enumerate(idx):
            part = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
            assign[group[i]["declaration_link"]] = part
    return assign


def split_temporal(rows, ratios=RATIOS):
    """Chronological split: oldest -> train, newest -> test. Undated rows go last."""
    ordered = sorted(rows, key=lambda r: (_parse_date(r["review_date"]) is None, _parse_date(r["review_date"]) or (0, 0, 0)))
    n_train, n_val, _ = _assign_counts(len(ordered), ratios)
    assign = {}
    for rank, r in enumerate(ordered):
        part = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
        assign[r["declaration_link"]] = part
    return assign


def split_speaker(rows, seed=42, ratios=RATIOS):
    """Speaker-disjoint split: all statements of a speaker land in the same split.
    Speakers are shuffled and greedily packed to approximate the row ratios."""
    rng = random.Random(seed)
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[_speaker_key(r)].append(r)
    speakers = list(by_speaker)
    rng.shuffle(speakers)
    total = len(rows)
    target_train, target_val = total * ratios[0], total * ratios[1]
    assign = {}
    n_train = n_val = 0
    for sp in speakers:
        group = by_speaker[sp]
        if n_train < target_train:
            part = "train"; n_train += len(group)
        elif n_val < target_val:
            part = "val"; n_val += len(group)
        else:
            part = "test"
        for r in group:
            assign[r["declaration_link"]] = part
    return assign


SPLITTERS = {
    "random": split_random,
    "temporal": split_temporal,
    "speaker": split_speaker,
}


def make_split(rows, protocol="random", seed=42):
    """Dispatch to a protocol; `seed` is ignored by the deterministic temporal one."""
    fn = SPLITTERS[protocol]
    if protocol == "temporal":
        return fn(rows)
    return fn(rows, seed=seed)
