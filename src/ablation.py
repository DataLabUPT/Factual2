"""Ablation / segmentation runner for Factual2.

Trains a TF-IDF + Logistic Regression baseline (the light, GPU-free baseline used
for reference in the LIAR2 study) over the generated grid and reports accuracy and
macro-F1. It reads the split files produced by build_dataset.py:

    data/splits/<scheme>/<protocol>/<construction>/{train,val,test}.csv

Ablation axes (A5, the encoder arm, is outside this release):
  A1 feature group : text | credibility | text+credibility
  A2 construction  : psa | wsa | lfa | lfa_temporal   (credibility source)
  A3 protocol      : random | temporal | speaker
  A4 credibility columns : full vector | total_counts only | each bucket individually
                           (--cred-cols full,total,individual)
  A6 context field : --context off|on|both — whether the `context` text is folded into
                     the text features (--with-context is an alias for on)
With --cred-repr, the runner instead sweeps the *representation* of the credibility
vector: the released counts against length-invariant
alternatives, to test whether the LFA/LFA-temporal gap under the temporal protocol is an
artifact of feature scale rather than the price of causal admissibility. The released
split files are untouched -- the transform is applied after loading, so no new leakage
surface is created. Modes: counts (released: per-class buckets + total_counts),
proportions (buckets divided by the total, zero vector preserved for cold start),
proportions+loglen (proportions plus log(1+total) as an explicit history-length channel),
counts+loglen (raw buckets plus log(1+total)). Output goes to results/cred_repr.csv, so
results/results.csv keeps its released schema.

Results are averaged over training seeds (--seeds) and written to
results/results.csv, with a printed summary of the headline comparison.

Usage:
    python src/ablation.py [--schemes 5c,3c,2p,2c] [--protocols random,temporal,speaker]
        [--constructions lfa,lfa_temporal,psa,wsa] [--groups text,credibility,text+credibility]
        [--cred-cols full,total,individual] [--context off,on,both] [--seeds 7,13,42,77,123]
"""
import csv
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
csv.field_size_limit(10 ** 7)


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load(scheme, protocol_tag, construction):
    base = os.path.join(SPLITS_DIR, scheme, protocol_tag, construction)
    parts = {}
    for part in ("train", "val", "test"):
        with open(os.path.join(base, f"{part}.csv"), encoding="utf-8") as f:
            parts[part] = list(csv.DictReader(f))
    cred_cols = [c for c in parts["train"][0] if c.endswith("_counts")]
    return parts, cred_cols


def _text(rows, with_context):
    if with_context:
        return [f"{r['content']} {r['context']}" for r in rows]
    return [r["content"] for r in rows]


CRED_REPRS = ("counts", "proportions", "proportions+loglen", "counts+loglen")


def _cred_matrix(rows, cred_cols, repr_mode):
    """Dense credibility block under one representation; cold-start rows stay all-zero."""
    class_cols = [c for c in cred_cols if c != "total_counts"]
    raw = np.array([[float(r[c]) for c in class_cols] for r in rows])
    tot = raw.sum(axis=1, keepdims=True)
    if repr_mode == "counts":
        return np.hstack([raw, tot])
    nz = np.where(tot > 0, tot, 1.0)
    if repr_mode == "proportions":
        return raw / nz
    if repr_mode == "proportions+loglen":
        return np.hstack([raw / nz, np.log1p(tot)])
    if repr_mode == "counts+loglen":
        return np.hstack([raw, np.log1p(tot)])
    raise SystemExit("unknown --cred-repr mode: %r" % (repr_mode,))


def _features(parts, cred_cols, group, with_context, seed, cred_repr="counts"):
    """Builds train/val/test feature matrices for a feature group."""
    tr, va, te = parts["train"], parts["val"], parts["test"]
    blocks = {"train": [], "val": [], "test": []}
    if "text" in group:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
        blocks["train"].append(vec.fit_transform(_text(tr, with_context)))
        blocks["val"].append(vec.transform(_text(va, with_context)))
        blocks["test"].append(vec.transform(_text(te, with_context)))
    if "credibility" in group:
        scaler = StandardScaler()

        def dense(rows):
            if cred_repr == "counts":
                return np.array([[float(r[c]) for c in cred_cols] for r in rows])
            return _cred_matrix(rows, cred_cols, cred_repr)

        blocks["train"].append(csr_matrix(scaler.fit_transform(dense(tr))))
        for part in ("val", "test"):
            blocks[part].append(csr_matrix(scaler.transform(dense(parts[part]))))
    X = {p: hstack(b).tocsr() for p, b in blocks.items()}
    y = {p: np.array([int(r["label"]) for r in parts[p]]) for p in ("train", "val", "test")}
    return X, y


def run_one(parts, cred_cols, group, with_context, seed, cred_repr="counts"):
    X, y = _features(parts, cred_cols, group, with_context, seed, cred_repr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X["train"], y["train"])
    out = {}
    for part in ("val", "test"):
        pred = clf.predict(X[part])
        out[f"{part}_acc"] = accuracy_score(y[part], pred)
        out[f"{part}_macrof1"] = f1_score(y[part], pred, average="macro")
    return out


GROUP_MAP = {"text": ("text",), "credibility": ("credibility",),
             "text+credibility": ("text", "credibility")}


def cred_variants(class_cols, modes):
    """A4 credibility-column subsets: yields (label, columns) per requested mode.
      full       -> the per-class vector plus total_counts (default; = the old behavior)
      total      -> only total_counts
      individual -> each class bucket on its own (one variant per column)"""
    for mode in modes:
        if mode == "full":
            yield "full", class_cols + ["total_counts"]
        elif mode == "total":
            yield "total", ["total_counts"]
        elif mode == "individual":
            for c in class_cols:
                yield c, [c]
        else:
            raise SystemExit(f"unknown --cred-cols mode: {mode!r}")


def main():
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    constructions = _arg("--constructions", "lfa,lfa_temporal,psa,wsa").split(",")
    groups = _arg("--groups", "text,credibility,text+credibility").split(",")
    cred_modes = _arg("--cred-cols", "full").split(",")
    seeds = [int(s) for s in _arg("--seeds", "7,13,42,77,123").split(",")]
    seed42 = _arg("--seed", "42")
    # A6 context axis: off | on | both (comma-list). --with-context is an alias for "on".
    if "--context" in sys.argv:
        context_modes = _arg("--context", "off").split(",")
    elif "--with-context" in sys.argv:
        context_modes = ["on"]
    else:
        context_modes = ["off"]
    context_modes = ["on", "off"] if context_modes == ["both"] else context_modes

    def tag(p):
        return p if p == "temporal" else f"{p}_seed{seed42}"

    results = []
    for scheme in schemes:
        for protocol in protocols:
            for construction in constructions:
                parts, cred_cols = load(scheme, tag(protocol), construction)
                class_cols = [c for c in cred_cols if c != "total_counts"]
                for group in groups:
                    uses_cred = "credibility" in GROUP_MAP[group]
                    uses_text = "text" in GROUP_MAP[group]
                    # credibility source only matters when credibility is in the group
                    if not uses_cred and construction != constructions[0]:
                        continue
                    # A4: credibility-column subsets (only when cred used);
                    # A6: context on/off (only when text used).
                    cred_vars = list(cred_variants(class_cols, cred_modes)) if uses_cred else [("-", [])]
                    ctx_vars = context_modes if uses_text else ["off"]
                    for cred_label, cred_subset in cred_vars:
                        for ctx in ctx_vars:
                            per_seed = [run_one(parts, cred_subset, GROUP_MAP[group], ctx == "on", s)
                                        for s in seeds]
                            row = {"scheme": scheme, "protocol": protocol,
                                   "construction": construction if uses_cred else "-",
                                   "group": group, "cred_cols": cred_label if uses_cred else "-",
                                   "context": ctx if uses_text else "-"}
                            for metric in ("test_acc", "test_macrof1", "val_macrof1"):
                                vals = np.array([r[metric] for r in per_seed])
                                row[metric] = f"{vals.mean():.4f}"
                                row[metric + "_std"] = f"{vals.std():.4f}"
                            results.append(row)
                            print(f"{scheme:3} {protocol:8} {row['construction']:13} {group:16} "
                                  f"{row['cred_cols']:16} ctx={row['context']:3} "
                                  f"test_F1={row['test_macrof1']}±{row['test_macrof1_std']} acc={row['test_acc']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "results.csv")
    fields = ["scheme", "protocol", "construction", "group", "cred_cols", "context",
              "test_acc", "test_acc_std", "test_macrof1", "test_macrof1_std",
              "val_macrof1", "val_macrof1_std"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n{len(results)} runs -> {out_path}")


def main_cred_repr():
    """X4: representation sweep of the credibility vector (counts-only groups)."""
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "temporal,random,speaker").split(",")
    constructions = _arg("--constructions", "lfa,lfa_temporal").split(",")
    reprs = _arg("--cred-repr", ",".join(CRED_REPRS)).split(",")
    seeds = [int(x) for x in _arg("--seeds", "42").split(",")]
    seed42 = _arg("--seed", "42")

    rows = []
    for scheme in schemes:
        for protocol in protocols:
            tag = protocol if protocol == "temporal" else "%s_seed%s" % (protocol, seed42)
            for construction in constructions:
                parts, cred_cols = load(scheme, tag, construction)
                line = []
                for rep in reprs:
                    per_seed = [run_one(parts, cred_cols, ("credibility",), False, s, rep)
                                for s in seeds]
                    vals = np.array([r["test_macrof1"] for r in per_seed])
                    k = _cred_matrix(parts["train"][:1], cred_cols, rep).shape[1]
                    rows.append({"scheme": scheme, "protocol": protocol,
                                 "construction": construction, "cred_repr": rep,
                                 "n_features": k,
                                 "test_macrof1": "%.4f" % vals.mean(),
                                 "test_macrof1_std": "%.4f" % vals.std()})
                    line.append("%s=%.3f" % (rep, vals.mean()))
                print("%-3s %-9s %-13s " % (scheme, protocol, construction) + "  ".join(line))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cred_repr.csv")
    fields = ["scheme", "protocol", "construction", "cred_repr", "n_features",
              "test_macrof1", "test_macrof1_std"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n%d runs -> %s" % (len(rows), out_path))


if __name__ == "__main__":
    main_cred_repr() if "--cred-repr" in sys.argv else main()
