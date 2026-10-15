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

Reads the released LFA split files (labels are construction-independent) and reuses
segment._speaker_key. Writes results/baselines.csv.

    python src/baselines.py [--schemes 5c,3c,2p,2c] [--protocols random,temporal,speaker]
        [--seed 42]
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ablation import load, RESULTS_DIR  # shared split loader + results path
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


def main():
    schemes = _arg("--schemes", "5c,3c,2p,2c").split(",")
    protocols = _arg("--protocols", "random,temporal,speaker").split(",")
    seed = _arg("--seed", "42")

    def tag(p):
        return p if p == "temporal" else f"{p}_seed{seed}"

    rows = []
    for scheme in schemes:
        for protocol in protocols:
            parts, _ = load(scheme, tag(protocol), "lfa")
            seen = len({_speaker_key(r) for r in parts["train"]} &
                       {_speaker_key(r) for r in parts["test"]})
            n_test_spk = len({_speaker_key(r) for r in parts["test"]})
            mj, idf = majority_f1(parts), identity_f1(parts)
            rows.append({"scheme": scheme, "protocol": protocol,
                         "majority_macrof1": f"{mj:.4f}", "identity_macrof1": f"{idf:.4f}",
                         "test_speakers_seen_in_train": seen, "test_speakers_total": n_test_spk})
            print(f"{scheme:3} {protocol:9} majority={mj:.3f} identity={idf:.3f} "
                  f"(test speakers seen in train: {seen}/{n_test_spk})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baselines.csv")
    fields = ["scheme", "protocol", "majority_macrof1", "identity_macrof1",
              "test_speakers_seen_in_train", "test_speakers_total"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
