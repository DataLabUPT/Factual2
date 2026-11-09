"""Reference baselines for Factual2: majority-class and speaker-identity.

Two elementary, deterministic baselines that locate the credibility numbers of
ablation.py on an absolute scale and separate identity *memorization* from history
*semantics* (manuscript Section 6.2, Table `tab:baselines`):

  - majority : constant predictor of the most frequent TRAIN class.
  - identity : class-balanced logistic regression over a one-hot speaker indicator,
               no credibility counts. Speakers are indexed on the TRAIN split; an
               evaluation row whose speaker is unseen in train maps to the zero
               vector, so under the speaker-disjoint protocol identity collapses onto
               the majority floor exactly.

Two further columns calibrate the no-information level for the reference classifier
itself, because a class-balanced logistic regression does NOT reproduce the majority
rule when its features carry nothing (manuscript Section 6.2):

  - zerofeat  : the same classifier fitted on an all-zero count matrix of the released
                shape. This is what a cold-start LFA row actually gets, and it can land
                *below* the majority floor.
  - permnull  : label-permutation null over --permutations draws -- the real, released
                count vectors with the TRAIN labels shuffled, scored against the true
                test labels. Mean, SD and the 95th percentile give a calibrated
                threshold for the speaker-disjoint falsification test, replacing the
                majority floor as the reference point.

With --combined, it instead runs the factorial comparison of identity against history
(manuscript Section 6.2, Table `tab:fusion`): each of the seven feature-block
combinations over identity / LFA counts / TF-IDF text, on every scheme x protocol cell
and split seed, written to results/baselines_combined.csv. This isolates how much of the
credibility signal is speaker memorization and how much is verdict history, both on their
own and fused with the text. The text block and the classifier are the ones of
ablation.py, so the `text` and `text+lfa` columns reproduce results.csv exactly.

Reads the released LFA split files (labels are construction-independent) and reuses
segment._speaker_key. Writes results/baselines.csv.

    python src/baselines.py [--schemes 5c,3c,2p,2c] [--protocols random,temporal,speaker]
        [--seed 42] [--permutations 200]
    python src/baselines.py --combined [--schemes ...] [--protocols ...]
        [--seeds 7,13,42,77,123]
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ablation import load, _text, RESULTS_DIR  # shared split loader + text + results path
from segment import _speaker_key

csv.field_size_limit(10 ** 7)


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def majority_f1(parts):
    ytr = [int(r["label"]) for r in parts["train"]]
    yte = np.array([int(r["label"]) for r in parts["test"]])
    maj = Counter(ytr).most_common(1)[0][0]
    return f1_score(yte, np.full(len(yte), maj), average="macro")


def identity_f1(parts):
    tr, te = parts["train"], parts["test"]
    speakers = sorted({_speaker_key(r) for r in tr})
    idx = {s: i for i, s in enumerate(speakers)}

    def X(rows):
        m = np.zeros((len(rows), len(speakers)), dtype=np.float32)
        for i, r in enumerate(rows):
            j = idx.get(_speaker_key(r))
            if j is not None:
                m[i, j] = 1.0
        return m

    ytr = np.array([int(r["label"]) for r in tr])
    yte = np.array([int(r["label"]) for r in te])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X(tr), ytr)
    return f1_score(yte, clf.predict(X(te)), average="macro")


def _counts(rows, cred_cols):
    return np.array([[float(r[c]) for c in cred_cols] for r in rows])


def _fit_score(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average="macro")


def zerofeat_f1(parts, cred_cols):
    """The reference classifier on an all-zero count matrix: what cold start really scores."""
    ytr = np.array([int(r["label"]) for r in parts["train"]])
    yte = np.array([int(r["label"]) for r in parts["test"]])
    k = len(cred_cols)
    return _fit_score(np.zeros((len(ytr), k)), ytr, np.zeros((len(yte), k)), yte)


def permnull_f1(parts, cred_cols, n_perm, seed=42):
    """Label-permutation null: released count vectors, TRAIN labels shuffled."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(_counts(parts["train"], cred_cols))
    Xte = scaler.transform(_counts(parts["test"], cred_cols))
    ytr = np.array([int(r["label"]) for r in parts["train"]])
    yte = np.array([int(r["label"]) for r in parts["test"]])
    rng = np.random.default_rng(seed)
    scores = np.array([_fit_score(Xtr, rng.permutation(ytr), Xte, yte)
                       for _ in range(n_perm)])
    return scores.mean(), scores.std(ddof=1), float(np.percentile(scores, 95))


def main():
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    seed = _arg("--seed", "42")
    n_perm = int(_arg("--permutations", "200"))

    def tag(p):
        return p if p == "temporal" else f"{p}_seed{seed}"

    rows = []
    for scheme in schemes:
        for protocol in protocols:
            parts, cred_cols = load(scheme, tag(protocol), "lfa")
            seen = len({_speaker_key(r) for r in parts["train"]} &
                       {_speaker_key(r) for r in parts["test"]})
            n_test_spk = len({_speaker_key(r) for r in parts["test"]})
            mj, idf = majority_f1(parts), identity_f1(parts)
            zf = zerofeat_f1(parts, cred_cols)
            pm, psd, p95 = permnull_f1(parts, cred_cols, n_perm)
            rows.append({"scheme": scheme, "protocol": protocol,
                         "majority_macrof1": f"{mj:.4f}", "identity_macrof1": f"{idf:.4f}",
                         "zerofeat_macrof1": f"{zf:.4f}",
                         "permnull_mean_macrof1": f"{pm:.4f}",
                         "permnull_sd_macrof1": f"{psd:.4f}",
                         "permnull_p95_macrof1": f"{p95:.4f}",
                         "n_permutations": n_perm,
                         "test_speakers_seen_in_train": seen, "test_speakers_total": n_test_spk})
            print(f"{scheme:3} {protocol:9} majority={mj:.3f} identity={idf:.3f} "
                  f"zerofeat={zf:.3f} permnull={pm:.3f}+-{psd:.3f} p95={p95:.3f} "
                  f"(test speakers seen in train: {seen}/{n_test_spk})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baselines.csv")
    fields = ["scheme", "protocol", "majority_macrof1", "identity_macrof1",
              "zerofeat_macrof1", "permnull_mean_macrof1", "permnull_sd_macrof1",
              "permnull_p95_macrof1", "n_permutations",
              "test_speakers_seen_in_train", "test_speakers_total"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")


# --- X1: identity vs. history, alone and fused with text ---------------------------------

GROUPS = ("identity", "lfa", "identity+lfa",
          "text", "text+identity", "text+lfa", "text+identity+lfa")


def _identity_block(parts):
    """One-hot speaker indicator, indexed on TRAIN; an unseen speaker maps to zeros."""
    speakers = sorted({_speaker_key(r) for r in parts["train"]})
    idx = {s: i for i, s in enumerate(speakers)}
    out = {}
    for part, rows in parts.items():
        m = np.zeros((len(rows), len(speakers)), dtype=np.float32)
        for i, r in enumerate(rows):
            j = idx.get(_speaker_key(r))
            if j is not None:
                m[i, j] = 1.0
        out[part] = csr_matrix(m)
    return out


def _text_block(parts):
    """TF-IDF of the statement only (context off), as in ablation._features."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
    out = {"train": vec.fit_transform(_text(parts["train"], False))}
    for part in ("val", "test"):
        out[part] = vec.transform(_text(parts[part], False))
    return out


def _cred_block(parts, cred_cols):
    """Standardized full count vector (per-class buckets + total_counts)."""
    scaler = StandardScaler()

    def dense(rows):
        return np.array([[float(r[c]) for c in cred_cols] for r in rows])

    out = {"train": csr_matrix(scaler.fit_transform(dense(parts["train"])))}
    for part in ("val", "test"):
        out[part] = csr_matrix(scaler.transform(dense(parts[part])))
    return out


def combined_f1(parts, blocks, group):
    """Fits one feature-block combination; returns val/test macro-F1."""
    use = [blocks[name] for name in group.split("+")]
    X = {p: hstack([b[p] for b in use]).tocsr() for p in ("train", "val", "test")}
    y = {p: np.array([int(r["label"]) for r in parts[p]]) for p in ("train", "val", "test")}
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X["train"], y["train"])
    return {f"{part}_macrof1": f1_score(y[part], clf.predict(X[part]), average="macro")
            for part in ("val", "test")}


def main_combined():
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    seeds = _arg("--seeds", "7,13,42,77,123").split(",")

    rows = []
    for seed in seeds:
        for scheme in schemes:
            for protocol in protocols:
                # temporal is deterministic: the same split is scored under every seed
                tag = protocol if protocol == "temporal" else f"{protocol}_seed{seed}"
                parts, cred_cols = load(scheme, tag, "lfa")
                blocks = {"identity": _identity_block(parts),
                          "text": _text_block(parts),
                          "lfa": _cred_block(parts, cred_cols)}
                line = []
                for group in GROUPS:
                    sc = combined_f1(parts, blocks, group)
                    rows.append({"split_seed": seed, "scheme": scheme, "protocol": protocol,
                                 "group": group,
                                 "val_macrof1": f"{sc['val_macrof1']:.4f}",
                                 "test_macrof1": f"{sc['test_macrof1']:.4f}"})
                    line.append(f"{group}={sc['test_macrof1']:.3f}")
                print(f"seed {seed:>3} {scheme:3} {protocol:9} " + "  ".join(line))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baselines_combined.csv")
    fields = ["split_seed", "scheme", "protocol", "group", "val_macrof1", "test_macrof1"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main_combined() if "--combined" in sys.argv else main()
