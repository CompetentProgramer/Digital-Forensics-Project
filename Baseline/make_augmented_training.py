#!/usr/bin/env python3
"""
make_augmented_training.py
Builds a bigger training file for the random forest by adding NORMAL domains

Run from Baseline/:
    python3 make_augmented_training.py --palau ../data/test_combined_multiclass.csv.gz
"""
import argparse
import csv
import gzip
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dns_tunnel_detector import extract_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Add normal Palau domains to the Bubnov training file")
    ap.add_argument("--palau", required=True, help="Palau CSV or CSV.GZ (columns domain,class)")
    ap.add_argument("--train", default=os.path.join(HERE, "..", "ML", "dataset", "training.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "ML", "dataset", "training_augmented.csv"))
    ap.add_argument("--n-normal", type=int, default=20000, help="how many normal Palau domains to add")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.train, newline="", encoding="utf-8", errors="replace") as fh:
        bubnov = [(r[0].strip(), r[1].strip()) for r in csv.reader(fh) if len(r) >= 2 and r[1].strip()]
    seen = {d.lower().rstrip(".") for _, d in bubnov}

    opener = gzip.open if args.palau.endswith(".gz") else open
    normals = []
    with opener(args.palau, "rt", newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            d = (row["domain"] or "").strip().lower().rstrip(".")
            if row["class"].strip() == "0" and d and d not in seen:
                normals.append(d)

    rnd = random.Random(args.seed)
    n = min(args.n_normal, len(normals))
    added = rnd.sample(normals, n)
    with_sub = sum(extract_features(d, "A")["sub_length"] > 0 for d in added)

    rows = bubnov + [("0", d) for d in added]
    rnd.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    n_tun = sum(1 for lab, _ in rows if lab == "1")
    print(f"Bubnov rows: {len(bubnov)}  +  added normal Palau rows: {n} "
          f"({with_sub} of them have a subdomain)")
    print(f"Wrote {len(rows)} rows to {args.out}  ({n_tun} tunneling, {len(rows) - n_tun} normal)")


if __name__ == "__main__":
    main()