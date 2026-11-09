"""Per-class metrics and history-length strata for Factual2.

Two diagnostics of the reference classifier, both read off the same fitted models so that
they cost one pass:

  - per-class : precision, recall, F1 and test support for every class of a scheme,
                reported per split seed so that the variability of the rare verdicts is
                visible rather than hidden inside macro-F1.
  - strata    : the same models scored separately on test rows whose *admissible* history
                is empty, short (1-4 past verdicts) or substantial (>= 5). The stratum of
                a row is read from `total_counts` of the LFA file, which is by definition
                the history a leak-free feature is allowed to see, and the very same strata
                are then applied to the PSA predictions. PSA cannot be stratified by its
                own total, which counts the whole corpus; using the LFA strata for both is
                what makes the comparison meaningful. Accuracy is the primary per-stratum
                number, since macro-F1 over a few dozen rows is too unstable to carry a
                claim, and it is reported next to `majority_accuracy`, the train-majority
                constant predictor restricted to the same stratum: strata differ in class
                balance, so a bare accuracy cannot be read on its own.

The text-only group is kept as a control: it has no history, so a gradient across the
strata would indicate that the strata themselves differ in difficulty, while a flat text
line isolates the history effect. The feature blocks, the classifier and the split loader
are the ones of ablation.py, so the macro-F1 aggregated over these rows reproduces
results.csv.

Model seeds are inert for this pipeline (LR/lbfgs is deterministic on a fixed split), so
`--seeds` varies the *split* seed; the temporal protocol is deterministic and is therefore
scored once. Writes results/perclass.csv and results/strata.csv.

    python src/perclass_strata.py [--schemes 5c,3c,2c] [--protocols random,temporal,speaker]
        [--groups text,lfa,psa] [--seeds 7,13,42,77,123]
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ablation import load, _features, RESULTS_DIR
from schemes import SCHEMES

csv.field_size_limit(10 ** 7)

# (name, lowest total_counts, highest total_counts) -- no history / limited / substantial
STRATA = (("0", 0, 0), ("1-4", 1, 4), (">=5", 5, 10 ** 9))


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def class_names(scheme):
    """{integer class -> readable name}, merged classes joined with '+'."""
    out = {}
    for label, cls in SCHEMES[scheme].items():
        out.setdefault(cls, []).append(label)
    return {cls: "+".join(labels) for cls, labels in out.items()}


def _stratum(total):
    for name, lo, hi in STRATA:
        if lo <= total <= hi:
            return name
    raise SystemExit(f"total_counts out of range: {total!r}")


def _fit_predict(parts, cred_cols, group, seed):
    """Fits the reference classifier of ablation.py and returns (true, predicted) on test."""
    X, y = _features(parts, cred_cols, group, False, seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X["train"], y["train"])
    return y["test"], clf.predict(X["test"])


def main():
    schemes = _arg("--schemes", "5c,3c,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    groups = _arg("--groups", "text,lfa,psa").split(",")
    seeds = [int(x) for x in _arg("--seeds", "7,13,42,77,123").split(",")]

    per_rows, strat_rows = [], []
    for scheme in schemes:
        names = class_names(scheme)
        classes = sorted(names)
        for protocol in protocols:
            # temporal is a single deterministic split; model seeds change nothing
            for seed in ([42] if protocol == "temporal" else seeds):
                tag = protocol if protocol == "temporal" else f"{protocol}_seed{seed}"
                ref, _ = load(scheme, tag, "lfa")
                strata = [_stratum(float(r["total_counts"])) for r in ref["test"]]
                # the constant predictor of tab:baselines, restricted to each stratum: a
                # stratum's own class balance, without which per-stratum accuracy is unreadable
                maj = Counter(int(r["label"]) for r in ref["train"]).most_common(1)[0][0]
                for group in groups:
                    parts, cred_cols = load(scheme, tag, "psa" if group == "psa" else "lfa")
                    assert [r["declaration_link"] for r in parts["test"]] == \
                           [r["declaration_link"] for r in ref["test"]], \
                           f"test rows differ between lfa and {group} in {scheme}/{tag}"
                    block = ("text",) if group == "text" else ("credibility",)
                    yte, pred = _fit_predict(parts, cred_cols, block, seed)

                    prec, rec, f1, sup = precision_recall_fscore_support(
                        yte, pred, labels=classes, zero_division=0)
                    for i, cls in enumerate(classes):
                        per_rows.append({"scheme": scheme, "protocol": protocol,
                                         "split_seed": seed, "group": group,
                                         "class": cls, "class_name": names[cls],
                                         "test_support": int(sup[i]),
                                         "precision": f"{prec[i]:.4f}",
                                         "recall": f"{rec[i]:.4f}", "f1": f"{f1[i]:.4f}"})

                    for name, _lo, _hi in STRATA:
                        mask = np.array([s == name for s in strata])
                        if not mask.any():
                            continue
                        yb, pb = yte[mask], pred[mask]
                        strat_rows.append({"scheme": scheme, "protocol": protocol,
                                           "split_seed": seed, "group": group,
                                           "stratum": name, "n_test": int(mask.sum()),
                                           "majority_accuracy": f"{(yb == maj).mean():.4f}",
                                           "accuracy": f"{accuracy_score(yb, pb):.4f}",
                                           "macrof1": f"{f1_score(yb, pb, average='macro'):.4f}"})

                    line = "  ".join(
                        f"{r['stratum']}={r['accuracy']}({r['n_test']})"
                        for r in strat_rows if r["split_seed"] == seed and r["group"] == group
                        and r["scheme"] == scheme and r["protocol"] == protocol)
                    print(f"{scheme:3} {protocol:9} seed {seed:>3} {group:4} "
                          f"macroF1={f1_score(yte, pred, average='macro'):.3f}  acc by history: "
                          f"{line}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for rows, fname, fields in (
            (per_rows, "perclass.csv",
             ["scheme", "protocol", "split_seed", "group", "class", "class_name",
              "test_support", "precision", "recall", "f1"]),
            (strat_rows, "strata.csv",
             ["scheme", "protocol", "split_seed", "group", "stratum", "n_test",
              "majority_accuracy", "accuracy", "macrof1"])):
        out_path = os.path.join(RESULTS_DIR, fname)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
