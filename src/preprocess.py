"""Preprocess a scraped factual.ro CSV into the clean Factual2 base corpus.

Reconstructed (2026-07-06) from the retired preprocessing step so the base
corpus can be regenerated when the upstream extraction is updated. It is
split-agnostic: it renames columns to snake_case, drops the free-text
`check subject` / `conclusion` fields and the `photo` column, normalizes the
`label` verdict to the English vocabulary used downstream, and keeps only the
statements carrying one of the five verdicts. It deliberately does NOT emit
speaker-credibility count columns: those are leak-free constructions that depend
on the train/test split and are produced separately (build_dataset.py).

    python src/preprocess.py ["data/factual-ro 20260702.csv"]

By default the most recent `factual-ro *.csv` extraction in data/ is processed.
The result is written to data/factual2.csv (the canonical base corpus).
"""
import csv
import glob
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV = os.path.join(DATA_DIR, "factual2.csv")

csv.field_size_limit(10 ** 7)

# Output schema (snake_case), in the canonical factual2.csv column order.
BASE_FIELDS = ["content", "review_date", "author_name", "author_link", "affiliation",
               "source_name", "source_link", "context", "label", "declaration_link"]

# Raw (spaced) -> snake_case column names. Columns absent here (photo,
# check subject, conclusion) are dropped. The raw `date` is the date the
# verification was PUBLISHED (not when the statement was made), so it is
# released as `review_date`.
RENAME = {
    "content": "content", "date": "review_date",
    "author name": "author_name", "author link": "author_link",
    "affiliation": "affiliation",
    "source name": "source_name", "source link": "source_link",
    "context": "context", "label": "label",
    "declaration link": "declaration_link",
}

# Romanian verdict -> English; anything not here is not a verdict and is dropped.
LABEL_RO_EN = {
    "Fals": "false",
    "Adevărat": "true",
    "Trunchiat": "truncated",
    "Parțial Adevărat": "partially true",
    "Parțial Fals": "partially false",
}
CLASSES = ["false", "true", "truncated", "partially true", "partially false"]


def latest_csv():
    """Most recent `factual-ro *.csv` extraction in data/ (by date in filename)."""
    cands = glob.glob(os.path.join(DATA_DIR, "factual-ro*.csv"))

    def key(p):
        m = re.search(r"(\d{8})", os.path.basename(p))
        return m.group(1) if m else ""
    return max(cands, key=key) if cands else None


def english_label(raw_label):
    """Normalized English verdict ('' if not one of the five)."""
    return LABEL_RO_EN.get((raw_label or "").strip(), "")


def base_rows(src_path):
    """Read a scraped CSV; return clean base rows (verdict rows only)."""
    out = []
    with open(src_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = english_label(row.get("label"))
            if not label:
                continue
            rec = {RENAME[k]: (v or "") for k, v in row.items() if k in RENAME}
            rec["label"] = label
            out.append({k: rec.get(k, "") for k in BASE_FIELDS})
    return out


def preprocess(src_path, out_path=OUT_CSV):
    rows = base_rows(src_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Base corpus: {len(rows)} statements -> {out_path}")
    return rows


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else latest_csv()
    if not src or not os.path.isfile(src):
        sys.exit("Niciun fisier CSV de procesat in data/.")
    preprocess(src)
