"""Label schemes for Factual2.

Each scheme maps the five English base verdicts onto an integer-coded target and
selects which statements are kept (labels absent from the map are dropped). The
credibility count columns follow the scheme's classes, so a merged class carries
a single count bucket named after its representative verdict.

  5c  5-class ordinal (all rows):   false=0, partially false=1, truncated=2,
                                     partially true=3, true=4
  3c  3-class (all rows):           {false, partially false}=0, truncated=1,
                                     {partially true, true}=2
  2p  binary-partial (no truncated):{false, partially false}=0,
                                     {partially true, true}=1
  2c  binary strict (pure only):    false=0, true=1
"""

# scheme -> {english label -> integer class}; labels not present are dropped.
SCHEMES = {
    "5c": {"false": 0, "partially false": 1, "truncated": 2, "partially true": 3, "true": 4},
    "3c": {"false": 0, "partially false": 0, "truncated": 1, "partially true": 2, "true": 2},
    "2p": {"false": 0, "partially false": 0, "partially true": 1, "true": 1},
    "2c": {"false": 0, "true": 1},
}

# scheme -> {integer class -> count column name} (representative verdict per class).
COUNT_NAMES = {
    "5c": {0: "false_counts", 1: "partially_false_counts", 2: "truncated_counts",
           3: "partially_true_counts", 4: "true_counts"},
    "3c": {0: "false_counts", 1: "truncated_counts", 2: "true_counts"},
    "2p": {0: "false_counts", 1: "true_counts"},
    "2c": {0: "false_counts", 1: "true_counts"},
}


def classes(scheme):
    """Ordered list of integer classes for a scheme."""
    return sorted(set(SCHEMES[scheme].values()))


def apply_scheme(rows, scheme):
    """Return new rows kept by the scheme, with `label` recoded to its integer
    class. Original rows are left untouched."""
    mapping = SCHEMES[scheme]
    out = []
    for r in rows:
        if r["label"] in mapping:
            nr = dict(r)
            nr["label"] = mapping[r["label"]]
            out.append(nr)
    return out
