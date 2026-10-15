"""Build the released Factual2 splits: label schemes x segmentation x credibility.

Reads the clean base corpus data/factual2.csv, then: segment ONCE on the base (so a
statement keeps the same split across schemes) -> for each label scheme, recode
labels + drop out-of-scheme rows -> credibility count vectors over the scheme's
classes -> write per-split CSVs:

    data/splits/<scheme>/<protocol>[_seed<k>]/<construction>/{train,val,test}.csv

Row schema (count columns depend on the scheme):
    content, review_date, author_name, author_link, affiliation, source_name,
    source_link, context, label, <class count columns...>,
    total_counts, declaration_link

Usage:
    python src/build_dataset.py [csv_path]     # defaults to data/factual2.csv
        [--schemes 5c,3c,2p,2c] [--protocols random,temporal,speaker]
        [--constructions lfa,lfa_temporal,psa,wsa] [--seed 42]
"""
import csv
import os
import sys

import segment
import schemes as schemes_mod
import credibility

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
DEFAULT_CSV = os.path.join(DATA_DIR, "factual2.csv")
csv.field_size_limit(10 ** 7)

META = ["content", "review_date", "author_name", "author_link", "affiliation",
        "source_name", "source_link", "context"]


def base_rows(src_path):
    """Reads the clean base corpus (data/factual2.csv) into a list of dict rows.
    The corpus is already snake_case, English-labelled and verdict-only, so no
    transformation happens here — the scraping/preprocessing that produced it has
    been retired (see repo history)."""
    with open(src_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _fieldnames(scheme):
    cls = schemes_mod.classes(scheme)
    count_cols = [schemes_mod.COUNT_NAMES[scheme][c] for c in cls]
    return META + ["label"] + count_cols + ["total_counts", "declaration_link"]


def write_split(rows, splits, counts, fieldnames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        merged = dict(r)
        merged.update(counts[r["declaration_link"]])
        buckets[splits[r["declaration_link"]]].append(merged)
    for part, part_rows in buckets.items():
        with open(os.path.join(out_dir, f"{part}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(part_rows)
    return {k: len(v) for k, v in buckets.items()}


def build(src_path, scheme_names, protocols, constructions, seed=42):
    base = base_rows(src_path)
    for protocol in protocols:
        splits = segment.make_split(base, protocol=protocol, seed=seed)
        tag = protocol if protocol == "temporal" else f"{protocol}_seed{seed}"
        for scheme in scheme_names:
            rows = schemes_mod.apply_scheme(base, scheme)
            cls = schemes_mod.classes(scheme)
            names = schemes_mod.COUNT_NAMES[scheme]
            fields = _fieldnames(scheme)
            for construction in constructions:
                counts = credibility.credibility_columns(rows, splits, cls, names, construction)
                out_dir = os.path.join(SPLITS_DIR, scheme, tag, construction)
                sizes = write_split(rows, splits, counts, fields, out_dir)
                print(f"[{scheme}] {tag}/{construction:12}: {sizes}")


if __name__ == "__main__":
    positional = [a for i, a in enumerate(sys.argv[1:], 1)
                  if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    src = positional[0] if positional else DEFAULT_CSV
    scheme_names = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    constructions = _arg("--constructions", "lfa,lfa_temporal,psa,wsa").split(",")
    seed = int(_arg("--seed", "42"))
    if not src or not os.path.isfile(src):
        print("Niciun fisier CSV de procesat.")
        sys.exit(1)
    print(f"Sursa: {src}")
    build(src, scheme_names, protocols, constructions, seed=seed)
