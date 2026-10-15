"""Restore the statement texts into a text-free Factual2 release.

The public release ships every derived artifact (labels, speaker keys, dates,
split assignments, credibility count vectors, results) but NOT the two editorial
text fields of factual.ro — `content` (the statement) and `context`. This script
puts them back from a local copy of the factual.ro extraction, joining on
`declaration_link`, so the full pipeline (build_dataset.py -> ablation.py)
becomes runnable.

    python src/rehydrate.py --source "factual-ro 20260702.csv"
        [--data-dir data] [--splits] [--allow-partial]

The source CSV may be either the raw extraction (columns `declaration link`,
`content`, `context`) or any CSV carrying a URL column plus the text columns;
both snake_case and space-separated headers are accepted.

By default the script writes `<data-dir>/factual2.csv` — the canonical base
corpus — and stops there: `python src/build_dataset.py` then regenerates every
split file with text, byte-identically to the released ones (the splits are
deterministic given the seed). With --splits it instead re-attaches `content`
and `context` to the released split CSVs in place, leaving every other column,
and the row order, untouched.
"""
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_CSV = "factual2_metadata.csv"
BASE_CSV = "factual2.csv"

# Canonical base-corpus column order (see docs/SCHEMA.md).
BASE_FIELDS = ["content", "review_date", "author_name", "author_link", "affiliation",
               "source_name", "source_link", "context", "label", "declaration_link"]
TEXT_FIELDS = ["content", "context"]

csv.field_size_limit(10 ** 7)


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _norm(name):
    return (name or "").strip().lower().replace(" ", "_")


def text_index(source_path):
    """{declaration_link -> {content, context}} from a factual.ro extraction."""
    with open(source_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty source file: {source_path}")
        cols = {_norm(c): c for c in reader.fieldnames}
        missing = [c for c in ["declaration_link", "content"] if c not in cols]
        if missing:
            raise SystemExit(
                f"Source is missing column(s) {missing}; found: {reader.fieldnames}")
        index = {}
        for row in reader:
            url = (row[cols["declaration_link"]] or "").strip()
            if not url:
                continue
            index[url] = {
                "content": (row[cols["content"]] or "").strip(),
                "context": (row[cols["context"]] or "").strip() if "context" in cols else "",
            }
    return index


def _rehydrate_rows(rows, index, label):
    """Fills the text fields in place; returns the list of uncovered URLs."""
    missing = []
    for row in rows:
        text = index.get((row.get("declaration_link") or "").strip())
        if text is None or not text["content"]:
            missing.append(row.get("declaration_link", ""))
            for field in TEXT_FIELDS:
                row.setdefault(field, "")
            continue
        for field in TEXT_FIELDS:
            row[field] = text[field]
    print(f"    {label}: {len(rows) - len(missing)}/{len(rows)} rows rehydrated")
    return missing


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _write(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def rehydrate_base(data_dir, index, allow_partial=False):
    src = os.path.join(data_dir, METADATA_CSV)
    if not os.path.isfile(src):
        raise SystemExit(f"Not found: {src} (run this from the release root)")
    rows, _ = _read(src)
    missing = _rehydrate_rows(rows, index, METADATA_CSV)
    if missing and not allow_partial:
        raise SystemExit(
            f"{len(missing)} row(s) have no text in the source, e.g. {missing[:3]}.\n"
            "Use a complete extraction, or pass --allow-partial to write them empty.")
    out = os.path.join(data_dir, BASE_CSV)
    _write(out, rows, BASE_FIELDS)
    print(f"    wrote {out} ({len(rows)} rows)")
    return missing


def rehydrate_splits(data_dir, index, allow_partial=False):
    splits_dir = os.path.join(data_dir, "splits")
    if not os.path.isdir(splits_dir):
        raise SystemExit(f"Not found: {splits_dir}")
    total_missing = 0
    for root, _dirs, files in os.walk(splits_dir):
        for name in sorted(files):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(root, name)
            rows, fields = _read(path)
            rel = os.path.relpath(path, splits_dir).replace(os.sep, "/")
            total_missing += len(_rehydrate_rows(rows, index, rel))
            # text columns go back where build_dataset.py puts them: `content`
            # first, `context` just before `label`
            fields = [f for f in fields if f not in TEXT_FIELDS]
            anchor = fields.index("label") if "label" in fields else len(fields)
            fields = ["content"] + fields[:anchor] + ["context"] + fields[anchor:]
            _write(path, rows, fields)
    if total_missing and not allow_partial:
        raise SystemExit(f"{total_missing} split row(s) had no text in the source.")


if __name__ == "__main__":
    source = _arg("--source")
    if not source or not os.path.isfile(source):
        raise SystemExit(__doc__)
    data_dir = _arg("--data-dir", DEFAULT_DATA_DIR)
    allow_partial = "--allow-partial" in sys.argv
    print(f"Source: {source}")
    index = text_index(source)
    print(f"    {len(index)} statements indexed by declaration_link")
    if "--splits" in sys.argv:
        rehydrate_splits(data_dir, index, allow_partial)
    else:
        rehydrate_base(data_dir, index, allow_partial)
        print("\nNext: python src/build_dataset.py   # regenerates the splits with text")
