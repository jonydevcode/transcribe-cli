# SPDX-License-Identifier: Apache-2.0
"""Word error rate of a hypothesis transcript against a reference: wer.py REFERENCE HYPOTHESIS."""

import re
import sys
import unicodedata


def words(path, normalize):
    text = open(path, encoding="utf-8").read()
    if normalize:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[^\w\s']", " ", text)
    return text.split()

def counts(ref, hyp):
    # row[j] = (cost, substitutions, deletions, insertions) for ref[:i] vs hyp[:j]
    row = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, r in enumerate(ref, 1):
        new = [(i, 0, i, 0)]
        for j, h in enumerate(hyp, 1):
            if r == h:
                new.append(row[j - 1])
                continue
            c, s, d, n = row[j - 1]
            best = (c + 1, s + 1, d, n)
            c, s, d, n = row[j]
            best = min(best, (c + 1, s, d + 1, n))
            c, s, d, n = new[j - 1]
            new.append(min(best, (c + 1, s, d, n + 1)))
        row = new
    return row[-1]

ref_path, hyp_path = sys.argv[1:3]
for label, normalize in (("normalized (lowercase, no punctuation)", True), ("raw", False)):
    ref, hyp = words(ref_path, normalize), words(hyp_path, normalize)
    cost, s, d, i = counts(ref, hyp)
    print(f"{label}: WER {cost / len(ref):.2%}  ({s} sub, {d} del, {i} ins over {len(ref)} reference words)")
