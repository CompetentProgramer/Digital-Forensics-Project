#!/usr/bin/env python3
"""
cross_dataset_test.py
Train on one dataset, test on another: do the results survive a dataset made with different tools?

Run from Baseline/:
    python3 cross_dataset_test.py --palau path/to/test_combined_multiclass.csv.gz
"""
import argparse
import csv
import gzip
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(HERE, "..", "ML"))  # teammate's train.py and data_loader.py

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare_rf_vs_rules import load, rule_predict, score  # noqa: E402
from data_loader import features_from_domains  # noqa: E402  (ML/)
from train import load_model, predict, train_model  # noqa: E402  (ML/)
from dns_tunnel_detector import extract_features  # noqa: E402

TWO_PART_SUFFIXES = {"co.uk", "com.br", "com.au", "co.jp", "com.tr", "com.cn", "co.in", "co.za", "com.mx",
                     "com.ar", "org.uk", "ac.uk", "gov.uk", "com.tw", "co.kr", "com.ua", "com.pl", "co.nz"}


def norm(d):
    return d.strip().lower().rstrip(".")


def load_palau(path, domain_col, class_col):
    opener = gzip.open if path.endswith(".gz") else open
    domains, classes = [], []
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if domain_col not in reader.fieldnames or class_col not in reader.fieldnames:
            sys.exit(f"Columns not found. File has: {reader.fieldnames}. Use --domain-col / --class-col.")
        for row in reader:
            d = norm(row[domain_col] or "")
            if d:
                domains.append(d)
                classes.append(int(float(row[class_col])))
    return domains, np.array(classes)


def main():
    ap = argparse.ArgumentParser(description="Train on Bubnov, test on the Palau dataset")
    ap.add_argument("--palau", required=True, help="Palau CSV or CSV.GZ (columns domain,class)")
    ap.add_argument("--train", default=os.path.join(HERE, "..", "ML", "dataset", "training.csv"))
    ap.add_argument("--model", help="load this saved .joblib model instead of training (it must have been trained on --train)")
    ap.add_argument("--plot", default=os.path.join(HERE, "..", "Metrics", "cross_dataset.png"),
                    help="chart PNG (default: ../Metrics/cross_dataset.png)")
    ap.add_argument("--domain-col", default="domain")
    ap.add_argument("--class-col", default="class")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "Metrics", "cross_dataset_metrics.csv"),
                    help="results CSV (default: ../Metrics/cross_dataset_metrics.csv)")
    args = ap.parse_args()

    d_tr, y_tr = load(args.train)
    f_tr = [extract_features(d, "A") for d in d_tr]

    d_te, c_te = load_palau(args.palau, args.domain_col, args.class_col)
    seen = {norm(d) for d in d_tr}
    keep = np.array([d not in seen for d in d_te])
    removed = int((~keep).sum())
    d_te = [d for d, k in zip(d_te, keep) if k]
    c_te = c_te[keep]
    print(f"Train (Bubnov): {len(d_tr)} rows")
    print(f"Test (Palau): {len(d_te)} rows after removing {removed} domains also seen in Bubnov training")
    print(f"  normal {int((c_te == 0).sum())}, DGA {int((c_te == 1).sum())}, tunneling {int((c_te == 2).sum())}\n")

    f_te = [extract_features(d, "A") for d in d_te]
    X_te = features_from_domains(d_te)

    rf = load_model(args.model) if args.model else train_model(features_from_domains(d_tr), y_tr)
    rf_pred = predict(rf, X_te)

    best_t, best_f1 = 50, -1.0
    for t in range(10, 101, 2):
        f1 = score(y_tr, rule_predict(f_tr, t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    systems = [
        (f"Random forest (trained on {os.path.basename(args.train)})", rf_pred),
        ("Rule detector (default, cutoff 50)", rule_predict(f_te)),
        (f"Rule detector (cutoff tuned on Bubnov: {best_t})", rule_predict(f_te, best_t)),
    ]

    eval_idx = np.where(c_te != 1)[0]  # normal + tunneling
    y_eval = (c_te[eval_idx] == 2).astype(int)
    dga_idx = np.where(c_te == 1)[0]

    results = []
    print(f"{'system':<52}{'prec':>7}{'rec':>7}{'f1':>7}{'FP':>8}{'FN':>6}{'FP rate':>9}{'DGA flagged':>13}")
    for name, pred in systems:
        m = score(y_eval, pred[eval_idx])
        fpr = m["fp"] / max(m["fp"] + m["tn"], 1)
        dga_rate = float(pred[dga_idx].mean()) if len(dga_idx) else 0.0
        results.append((name, m, fpr, dga_rate))
        print(f"{name:<52}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['fp']:>8}"
              f"{m['fn']:>6}{fpr:>9.2%}{dga_rate:>13.1%}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "false_alarm_rate",
                    "dga_flagged_rate"])
        for name, m, fpr, dga_rate in results:
            w.writerow([name, round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4), m["tp"],
                        m["fp"], m["fn"], m["tn"], round(fpr, 5), round(dga_rate, 4)])
    print(f"\nWrote {args.out}")

    # --- Chart: grouped bars, same style as comparison.png -------------------
    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    metrics_order = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=surface)
    ax.set_facecolor(surface)
    width = 0.26
    xs = np.arange(len(metrics_order))
    for i, (name, m, fpr, dga_rate) in enumerate(results):
        vals = [m[k] for k, _ in metrics_order]
        bars = ax.bar(xs + (i - 1) * width, vals, width=width, color=colors[i], edgecolor=surface,
                      linewidth=2, label=f"{name}  [FP {m['fp']:,}, FN {m['fn']:,}]", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=ink)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in metrics_order], color=ink2, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=muted, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    n_norm, n_tun = int((c_te == 0).sum()), int((c_te == 2).sum())
    ax.set_title("Cross-dataset test: trained on Bubnov, tested on Palau", loc="left", fontsize=14,
                 color=ink, pad=34)
    ax.text(0, 1.045, f"Scored on {n_norm:,} normal and {n_tun:,} tunneling Palau queries "
                      "(DGA excluded). Scores are for the tunneling class; FP/FN are counts.",
            transform=ax.transAxes, fontsize=9.5, color=ink2, va="bottom")
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=1, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(ink2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig.savefig(args.plot, dpi=160, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.plot}")

    # --- Where do the systems fail? -----------------------------------------
    base = lambda d: ".".join(d.split(".")[-2:])
    tun = np.where(c_te == 2)[0]
    print("\nTunneling queries missed, by base domain (tunneling set has "
          f"{len(set(base(d_te[i]) for i in tun))} base domains):")
    for name, pred in systems:
        miss = [d_te[i] for i in tun if pred[i] == 0]
        print(f"  {name}: {len(miss)} of {len(tun)} missed  {Counter(base(d) for d in miss).most_common(2)}")
        for d in sorted(miss, key=len)[:2]:
            print(f"      e.g. ({len(d)} chars) {d}")

    # Does the two-part-suffix problem (co.uk, com.br ...) explain forest false alarms?
    norm_idx = np.where(c_te == 0)[0]
    two = np.array([base(d_te[i]) in TWO_PART_SUFFIXES for i in norm_idx])
    fp_rf = rf_pred[norm_idx] == 1
    print("\nRandom forest false alarms on normal domains:")
    print(f"  under common two-part suffixes (co.uk, com.br ...): {fp_rf[two].sum()} of {two.sum()} "
          f"({fp_rf[two].mean():.2%})")
    print(f"  all other normal domains:                          {fp_rf[~two].sum()} of {(~two).sum()} "
          f"({fp_rf[~two].mean():.2%})")
    fp_ex = [d_te[i] for i, f in zip(norm_idx, fp_rf) if f][:5]
    print(f"  examples: {fp_ex}")


if __name__ == "__main__":
    main()