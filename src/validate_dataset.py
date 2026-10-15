"""Dataset-quality validator for Factual2.

Runs three families of checks and exits non-zero if any HARD check fails:

  1. Integrity of the base corpus (data/factual2.csv): schema, unique non-empty
     declaration_link, non-empty content, valid verdicts, parseable/non-future
     dates, no content duplicated under conflicting labels.
  2. factual.ro slug <-> verdict consistency: where a declaration URL slug carries
     a verdict prefix, it must agree with the `label` (reported as a WARNING, since
     editorial slugs are advisory, not authoritative).
  3. Split & leakage correctness (data/splits/<scheme>/<protocol>/<construction>):
     splits disjoint and complete, label recoding correct, protocol invariants
     (speaker-disjoint / temporal ordering), and — the core property — the
     credibility count vectors match an INDEPENDENT re-derivation of each
     construction, so PSA leaks (includes self), WSA excludes self within split,
     and LFA / LFA-temporal are verified leak-free (train-only, no own label).

    python src/validate_dataset.py            # full report
    python src/validate_dataset.py --quiet    # only the summary + failures
"""
import csv
import datetime as dt
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import _speaker_key, _parse_date
from schemes import SCHEMES, COUNT_NAMES, classes as scheme_classes, apply_scheme

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
BASE_CSV = os.path.join(DATA_DIR, "factual2.csv")

BASE_COLS = ["content", "review_date", "author_name", "author_link", "affiliation",
             "source_name", "source_link", "context", "label", "declaration_link"]
VERDICTS = {"false", "true", "truncated", "partially true", "partially false"}
SLUG_PREFIX = [  # order matters: longer/partial prefixes first
    ("partial-adevarat-", "partially true"), ("partial-fals-", "partially false"),
    ("adevarat-", "true"), ("fals-", "false"), ("trunchiat-", "truncated"),
]
QUIET = "--quiet" in sys.argv

_fails, _warns = [], []


def fail(msg):
    _fails.append(msg)


def warn(msg):
    _warns.append(msg)


def info(msg):
    if not QUIET:
        print(msg)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- base corpus
def check_base():
    info("\n[1] Base corpus integrity (data/factual2.csv)")
    rows = _read(BASE_CSV)
    hdr = list(rows[0].keys()) if rows else []
    if hdr != BASE_COLS:
        fail(f"base: columns {hdr} != expected {BASE_COLS}")
    info(f"    {len(rows)} rows")

    links = Counter(r["declaration_link"].strip() for r in rows)
    if "" in links:
        fail(f"base: {links['']} rows have an empty declaration_link")
    dupes = {k: n for k, n in links.items() if k and n > 1}
    if dupes:
        fail(f"base: {len(dupes)} duplicated declaration_link(s), e.g. {list(dupes)[:3]}")
    else:
        info("    declaration_link: unique and non-empty  OK")

    empty_content = sum(1 for r in rows if not r["content"].strip())
    if empty_content:
        fail(f"base: {empty_content} rows have empty content")

    bad_label = sorted({r["label"] for r in rows if r["label"] not in VERDICTS})
    if bad_label:
        fail(f"base: labels outside the 5-verdict vocabulary: {bad_label}")
    else:
        info(f"    labels: {dict(Counter(r['label'] for r in rows))}")

    today = dt.date.today()
    bad_date = future = 0
    for r in rows:
        p = _parse_date(r["review_date"])
        if p is None:
            bad_date += 1
        else:
            y, m, d = p
            try:
                if dt.date(y, m, d) > today:
                    future += 1
            except ValueError:
                bad_date += 1
    if bad_date:
        fail(f"base: {bad_date} rows with unparseable date")
    if future:
        fail(f"base: {future} rows dated in the future (> {today})")
    if not bad_date and not future:
        info("    dates: all parseable and not in the future  OK")

    # content duplicated under conflicting labels = real contradiction
    by_content = defaultdict(set)
    for r in rows:
        by_content[r["content"].strip()].add(r["label"])
    conflict = {c: v for c, v in by_content.items() if c and len(v) > 1}
    if conflict:
        fail(f"base: {len(conflict)} identical statements with conflicting labels, "
             f"e.g. {list(conflict.items())[0][1]}")
    else:
        info("    no identical content under conflicting labels  OK")

    authorless = sum(1 for r in rows if not _speaker_key(r))
    info(f"    author-less rows: {authorless} (INFO)")
    return rows


# ------------------------------------------------------- slug <-> verdict
def check_slug(rows):
    info("\n[2] factual.ro slug <-> verdict consistency")

    def slug(u):
        m = re.search(r"/declaratii/([^/]+)/?$", u or "")
        return m.group(1) if m else ""

    have = match = 0
    mism = []
    for r in rows:
        s = slug(r["declaration_link"])
        v = next((lab for pre, lab in SLUG_PREFIX if s.startswith(pre)), None)
        if v:
            have += 1
            if v == r["label"]:
                match += 1
            else:
                mism.append((s[:48], v, r["label"]))
    info(f"    {have}/{len(rows)} rows carry a verdict prefix; {match} match")
    if mism:
        warn(f"slug: {len(mism)} slug/label mismatches (review): "
             + "; ".join(f"{s} -> slug={v}/label={l}" for s, v, l in mism[:5]))
    else:
        info("    all recognizable slugs agree with the label  OK")


# ------------------------------------------------------- splits & leakage
def _protocol_of(dirname):
    return dirname.split("_")[0]  # random_seed42 -> random


def _expected_cols(scheme):
    counts = [COUNT_NAMES[scheme][c] for c in scheme_classes(scheme)]
    return BASE_COLS[:-1] + counts + ["total_counts"] + [BASE_COLS[-1]]


def _reconstruct(construction, cls, splitmap, rows_by_link, order):
    """Independent re-derivation of the per-class count vector for every link,
    from the split assignment + integer labels + dates. Mirrors the leak-free
    DEFINITIONS (not credibility.py), so a match certifies the shipped columns."""
    link_speaker = {l: _speaker_key(r) for l, r in rows_by_link.items()}
    link_label = {l: int(r["label"]) for l, r in rows_by_link.items()}
    link_date = {l: _parse_date(r["review_date"]) for l, r in rows_by_link.items()}

    def counts_over(links):
        t = defaultdict(lambda: {c: 0 for c in cls})
        for l in links:
            t[link_speaker[l]][link_label[l]] += 1
        return t

    all_links = list(rows_by_link)
    train_links = [l for l in all_links if splitmap[l] == "train"]
    exp = {}
    if construction == "psa":
        glob = counts_over(all_links)
        for l in all_links:
            exp[l] = dict(glob[link_speaker[l]])
    elif construction == "wsa":
        per = {s: counts_over([l for l in all_links if splitmap[l] == s])
               for s in ("train", "val", "test")}
        for l in all_links:
            v = dict(per[splitmap[l]][link_speaker[l]])
            v[link_label[l]] = max(0, v[link_label[l]] - 1)
            exp[l] = v
    elif construction == "lfa":
        tr = counts_over(train_links)
        for l in all_links:
            sp = link_speaker[l]
            if splitmap[l] == "train":
                v = dict(tr[sp]); v[link_label[l]] = max(0, v[link_label[l]] - 1)
            elif sp in tr:
                v = dict(tr[sp])
            else:
                v = {c: 0 for c in cls}
            exp[l] = v
    elif construction == "lfa_temporal":
        train_by_sp = defaultdict(list)
        for l in train_links:
            train_by_sp[link_speaker[l]].append(l)
        for l in all_links:
            sp, d = link_speaker[l], link_date[l]
            v = {c: 0 for c in cls}
            if d is not None:
                for hl in train_by_sp.get(sp, []):
                    hd = link_date[hl]
                    if hd is not None and hd < d:
                        v[link_label[hl]] += 1
            exp[l] = v
    return exp


def check_splits(base_rows):
    info("\n[3] Split & leakage correctness (data/splits/)")
    base_label = {r["declaration_link"]: r["label"] for r in base_rows}  # original strings
    schemes = sorted(d for d in os.listdir(SPLITS_DIR)
                     if os.path.isdir(os.path.join(SPLITS_DIR, d)))
    checked = 0
    for scheme in schemes:
        kept = apply_scheme(base_rows, scheme)
        kept_links = {r["declaration_link"] for r in kept}
        cls = scheme_classes(scheme)
        cname = COUNT_NAMES[scheme]
        sdir = os.path.join(SPLITS_DIR, scheme)
        for proto_dir in sorted(os.listdir(sdir)):
            pdir = os.path.join(sdir, proto_dir)
            if not os.path.isdir(pdir):
                continue
            proto = _protocol_of(proto_dir)
            partitions = {}
            for constr in sorted(os.listdir(pdir)):
                cdir = os.path.join(pdir, constr)
                if not os.path.isdir(cdir):
                    continue
                tag = f"{scheme}/{proto_dir}/{constr}"
                parts = {}
                ok = True
                for split in ("train", "val", "test"):
                    fp = os.path.join(cdir, f"{split}.csv")
                    if not os.path.isfile(fp):
                        fail(f"{tag}: missing {split}.csv"); ok = False; continue
                    parts[split] = _read(fp)
                if not ok:
                    continue

                # columns
                exp_cols = _expected_cols(scheme)
                got = list(parts["train"][0].keys()) if parts["train"] else []
                if got != exp_cols:
                    fail(f"{tag}: columns {got} != expected {exp_cols}")

                linksets = {s: {r["declaration_link"] for r in rs} for s, rs in parts.items()}
                # disjoint
                if linksets["train"] & linksets["val"] or linksets["train"] & linksets["test"] \
                        or linksets["val"] & linksets["test"]:
                    fail(f"{tag}: splits are not disjoint")
                union = set().union(*linksets.values())
                if union != kept_links:
                    miss, extra = kept_links - union, union - kept_links
                    fail(f"{tag}: split union != scheme rows (missing {len(miss)}, extra {len(extra)})")

                rows_by_link = {r["declaration_link"]: r for rs in parts.values() for r in rs}
                splitmap = {l: s for s, ls in linksets.items() for l in ls}

                # label recoding correct + total == sum of buckets
                for r in rows_by_link.values():
                    want = SCHEMES[scheme][base_label[r["declaration_link"]]]
                    if int(r["label"]) != want:
                        fail(f"{tag}: bad label recoding for {r['declaration_link']}"); break
                    s = sum(int(r[cname[c]]) for c in cls)
                    if int(r["total_counts"]) != s:
                        fail(f"{tag}: total_counts != sum of buckets for {r['declaration_link']}"); break

                # protocol invariants
                if proto == "speaker":
                    sp_tr = {_speaker_key(r) for r in parts["train"]}
                    sp_ev = {_speaker_key(r) for r in parts["val"] + parts["test"]}
                    if sp_tr & sp_ev:
                        fail(f"{tag}: {len(sp_tr & sp_ev)} speaker(s) in both train and eval")
                if proto == "temporal":
                    td = [d for d in (_parse_date(r["review_date"]) for r in parts["train"]) if d]
                    ed = [d for d in (_parse_date(r["review_date"]) for r in parts["test"]) if d]
                    if td and ed and max(td) > min(ed):
                        fail(f"{tag}: temporal order violated (max train date > min test date)")

                # core: credibility vectors match an independent re-derivation
                exp = _reconstruct(constr, cls, splitmap, rows_by_link, None)
                bad = 0
                for l, r in rows_by_link.items():
                    for c in cls:
                        if int(r[cname[c]]) != exp[l][c]:
                            bad += 1; break
                if bad:
                    fail(f"{tag}: {bad} rows whose {constr} counts differ from the "
                         f"independent re-derivation (leak/definition mismatch)")

                # explicit leak-free assertion for LFA: all val/test rows of a
                # train-seen speaker share ONE vector; own label cannot enter.
                if constr == "lfa":
                    per_sp = defaultdict(set)
                    for r in parts["val"] + parts["test"]:
                        per_sp[_speaker_key(r)].add(tuple(int(r[cname[c]]) for c in cls))
                    if any(len(v) > 1 for v in per_sp.values()):
                        fail(f"{tag}: LFA eval vectors vary within a speaker (own-label leak)")

                partitions[constr] = linksets
                checked += 1

            # segmentation-once: all constructions share the same partition
            if len({frozenset((s, frozenset(ls)) for s, ls in p.items())
                    for p in partitions.values()}) > 1:
                fail(f"{scheme}/{proto_dir}: constructions disagree on the split partition")
    info(f"    checked {checked} scheme/protocol/construction combinations")


def main():
    if not os.path.isfile(BASE_CSV):
        sys.exit(f"missing {BASE_CSV}\nIn the public release the statement texts are not redistributed; restore the\nbase corpus first with: python src/rehydrate.py --source <factual.ro CSV>")
    base = check_base()
    check_slug(base)
    check_splits(base)

    print("\n" + "=" * 60)
    if _warns:
        print(f"WARNINGS ({len(_warns)}):")
        for w in _warns:
            print("  ! " + w)
    if _fails:
        print(f"FAILED ({len(_fails)} hard check(s)):")
        for f in _fails:
            print("  X " + f)
        sys.exit(1)
    print("PASSED — all hard checks OK" + (f" ({len(_warns)} warning(s))" if _warns else ""))


if __name__ == "__main__":
    main()
