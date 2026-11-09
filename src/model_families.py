"""Model-family robustness of the leakage gap for Factual2.

The reference classifier of ablation.py is a single linear model, so a reader can ask
whether the PSA-over-LFA inflation is a property of the *data* or of logistic regression.
This script answers that question directly: it re-runs the counts-only comparison of
Table 6 (PSA vs. LFA, full count vector) with seven classifiers that differ in inductive
bias -- a log-loss linear model, an RBF kernel, a nearest-neighbour rule, a single tree, a
bagged and a boosted tree ensemble, and a small MLP -- on exactly the released split
files, and reports the per-cell gap.

Only the classifier changes. The features are the shipped count columns of the split
files (`*_counts`, including `total_counts`), the splits are the released ones, and the
metric is test macro-F1, so the LogisticRegression row reproduces results.csv.

Scaling is per model, not per cell: the distance- and gradient-based models read
standardized counts (train statistics, as in ablation.py) and the trees read the raw
counts, axis-aligned splits being scale-invariant. kNN and the MLP have no class-weighting
option, so their absolute macro-F1 is lower on the imbalanced schemes; the quantity of
interest here is the PSA-minus-LFA difference within a model, not the level across models.

Model randomness is fixed (random_state=42) and the reported variability is over the
*split* seeds, the only source of variance that matters for this pipeline (see the
docstring of ablation.py). The temporal protocol is deterministic and is scored once.

Writes results/model_families.csv (one row per scheme x protocol x split seed x model)
and prints the gap table aggregated over the split seeds.

    python src/model_families.py [--schemes 5c,3c,2p,2c] [--protocols random,temporal,speaker]
        [--seeds 7,13,42,77,123]
"""
import csv
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ablation import load, RESULTS_DIR

csv.field_size_limit(10 ** 7)

# (label, factory, feature scaling) -- "std" = standardized counts, "raw" = shipped counts
MODELS = (
    ("LogisticRegression",
     lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                random_state=42), "std"),
    ("SVC (RBF)",
     lambda: SVC(kernel="rbf", class_weight="balanced", random_state=42), "std"),
    ("kNN (k=15)",
     lambda: KNeighborsClassifier(n_neighbors=15), "std"),
    ("DecisionTree",
     lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42), "raw"),
    ("RandomForest",
     lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42, n_jobs=-1), "raw"),
    ("HistGradientBoosting",
     lambda: HistGradientBoostingClassifier(random_state=42), "raw"),
    ("MLP (1x32)",
     lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=42), "std"),
)


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _blocks(parts, cred_cols, scaling):
    """Train/test count matrices under one scaling, plus the integer labels."""
    tr = np.array([[float(r[c]) for c in cred_cols] for r in parts["train"]])
    te = np.array([[float(r[c]) for c in cred_cols] for r in parts["test"]])
    if scaling == "std":
        scaler = StandardScaler().fit(tr)
        tr, te = scaler.transform(tr), scaler.transform(te)
    ytr = np.array([int(r["label"]) for r in parts["train"]])
    yte = np.array([int(r["label"]) for r in parts["test"]])
    return tr, ytr, te, yte


def _score(parts, cred_cols, factory, scaling):
    tr, ytr, te, yte = _blocks(parts, cred_cols, scaling)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf = factory().fit(tr, ytr)
    return f1_score(yte, clf.predict(te), average="macro")


def main():
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    seeds = [int(s) for s in _arg("--seeds", "7,13,42,77,123").split(",")]

    rows = []
    for scheme in schemes:
        for protocol in protocols:
            tags = ([("temporal", None)] if protocol == "temporal"
                    else [(f"{protocol}_seed{s}", s) for s in seeds])
            for tag, seed in tags:
                loaded = {c: load(scheme, tag, c) for c in ("lfa", "psa")}
                for name, factory, scaling in MODELS:
                    scores = {c: _score(p, cols, factory, scaling)
                              for c, (p, cols) in loaded.items()}
                    rows.append({"scheme": scheme, "protocol": protocol,
                                 "split_seed": seed if seed is not None else "",
                                 "model": name, "scaling": scaling,
                                 "lfa_macrof1": f"{scores['lfa']:.4f}",
                                 "psa_macrof1": f"{scores['psa']:.4f}",
                                 "gap_macrof1": f"{scores['psa'] - scores['lfa']:.4f}"})
                print(f"{scheme:3s} {protocol:9s} seed={str(seed or '-'):4s} done", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "model_families.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "protocol", "split_seed", "model",
                                          "scaling", "lfa_macrof1", "psa_macrof1",
                                          "gap_macrof1"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    # aggregated gap table: one line per model, one column per scheme x protocol cell
    cells = [(s, p) for s in schemes for p in protocols]
    header = "  ".join(f"{s}/{p[:4]:>5s}" for s, p in cells)
    print(f"{'model':22s} {header}")
    n_pos = n_tot = 0
    for name, _, _ in MODELS:
        line = []
        for scheme, protocol in cells:
            g = [float(r["gap_macrof1"]) for r in rows
                 if r["model"] == name and r["scheme"] == scheme and r["protocol"] == protocol]
            n_tot += 1
            n_pos += int(np.mean(g) > 0)
            line.append(f"{np.mean(g):+.3f}" + (f"({np.std(g, ddof=1):.3f})" if len(g) > 1 else "       "))
        print(f"{name:22s} " + "  ".join(line))
    print(f"\ngap > 0 in {n_pos}/{n_tot} model x scheme x protocol cells "
          f"(means over split seeds {seeds})")


if __name__ == "__main__":
    main()
