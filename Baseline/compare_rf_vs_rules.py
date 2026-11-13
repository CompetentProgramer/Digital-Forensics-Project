#!/usr/bin/env python3
"""
compare_rf_vs_rules.py
Head-to-head: the teammate's random forest (ML/) vs the base rule-based
detector (Baseline/dns_tunnel_detector.py), scored on the SAME held-out data.

    python3 compare_rf_vs_rules.py
    python3 compare_rf_vs_rules.py --train path/training.csv --valid path/validating.csv

Outputs: comparison_metrics.csv, comparison.png, plus an error breakdown.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # find dns_tunnel_detector next to this file

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402

from dns_tunnel_detector import apply_rules, extract_features  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "ML", "dataset")
RF_FEATURES = ["sub_length", "sub_entropy", "num_labels", "max_label_len", "digit_ratio"]  # as in ML/data_loader.py

class Thresholds:
    """Same defaults as dns_tunnel_detector.py's command-line flags."""
    def __init__(self, length=50):
        self.entropy, self.length, self.label, self.labels = 3.5, length, 30, 6

def load(path):
    """Bubnov files have no header: label,domain."""
    labels, domains = [], []

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip() in ("0", "1") and row[1].strip():
                labels.append(int(row[0]))
                domains.append(row[1].strip())
    return domains, np.array(labels)

def score(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": float(p[0]), "recall": float(r[0]),
            "f1": float(f[0]), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def rule_predict(feats, length_cutoff=50):
    t = Thresholds(length_cutoff)
    return np.array([1 if apply_rules(f, t) else 0 for f in feats])

def main():
    ap = argparse.ArgumentParser(description="Random forest vs rule-based detector on the same data")
    ap.add_argument("--train", default=os.path.join(DATA_DIR, "training.csv"))
    ap.add_argument("--valid", default=os.path.join(DATA_DIR, "validating.csv"))
    ap.add_argument("--out", default="comparison_metrics.csv")
    ap.add_argument("--plot", default="comparison.png")
    args = ap.parse_args()

    d_tr, y_tr = load(args.train)
    d_va, y_va = load(args.valid)
    f_tr = [extract_features(d, "A") for d in d_tr]
    f_va = [extract_features(d, "A") for d in d_va]

    print(f"Train: {len(y_tr)} rows   Validation: {len(y_va)} rows " f"({int(y_va.sum())} tunneling, {int((1 - y_va).sum())} normal)\n")

    # --- Random forest ---------------------------------
    X_tr = np.array([[f[k] for k in RF_FEATURES] for f in f_tr])
    X_va = np.array([[f[k] for k in RF_FEATURES] for f in f_va])
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    rf_pred = rf.predict(X_va)

    # --- Rule detector as-is ----------------------------------------------
    rule_pred = rule_predict(f_va)

    # --- Rules with length cutoff tuned on TRAIN only ----------------------
    best_t, best_f1 = 50, -1.0
    for t in range(10, 101, 2):
        f1 = score(y_tr, rule_predict(f_tr, t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    tuned_pred = rule_predict(f_va, best_t)

    systems = [
        ("Random forest ML", rf_pred),
        ("Rule detector (default, cutoff 50)", rule_pred),
        (f"Rule detector (cutoff tuned on train: {best_t})", tuned_pred),
    ]
    results = [(name, score(y_va, p)) for name, p in systems]

    print(f"{'system':<50}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'FP':>6}{'FN':>6}")
    for name, m in results:
        print(f"{name:<50}{m['accuracy']:>7.4f}{m['precision']:>7.4f}{m['recall']:>7.4f}" f"{m['f1']:>7.4f}{m['fp']:>6}{m['fn']:>6}")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])
        for name, m in results:
            w.writerow([name] + [round(m[k], 4) if isinstance(m[k], float) else m[k]
                                 for k in ("accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn")])
    print(f"\nWrote {args.out}")

    # --- Error breakdown -----------------------------------------------------
    print("\nRandom forest feature importance:")
    for k, v in sorted(zip(RF_FEATURES, rf.feature_importances_), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:.3f}")

    wrong = [(d, int(y)) for d, y, p in zip(d_va, y_va, rf_pred) if p != y]
    print(f"\nRandom forest mistakes on validation: {len(wrong)}")

    for d, y in wrong[:5]:
        print(f"  true={'tunneling' if y else 'normal'}  {d[:80]}")

    miss = [(len(d), d) for d, y, p in zip(d_va, y_va, rule_pred) if y == 1 and p == 0]
    caught_len = [len(d) for d, y, p in zip(d_va, y_va, rule_pred) if y == 1 and p == 1]

    print(f"\nTunneling queries the default rules miss: {len(miss)} of {int(y_va.sum())}")

    if miss:
        lens = [n for n, _ in miss]
        print(f"  missed length: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}; "
              f"caught length: min {min(caught_len)}, median {int(np.median(caught_len))}")
        
        for n, d in sorted(miss)[:3]:
            print(f"  e.g. ({n} chars) {d}")

    fp_rules = [d for d, y, p in zip(d_va, y_va, rule_pred) if y == 0 and p == 1]
    print(f"Normal queries the default rules flag: {len(fp_rules)}")

    # --- Chart: grouped bars, one group per metric ---------------------------
    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
    metrics_order = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=surface)
    ax.set_facecolor(surface)
    width = 0.26
    xs = np.arange(len(metrics_order))

    for i, (name, m) in enumerate(results):
        vals = [m[k] for k, _ in metrics_order]
        pos = xs + (i - 1) * width
        bars = ax.bar(pos, vals, width=width, color=colors[i], edgecolor=surface, linewidth=2, label=f"{name}  [FP {m['fp']}, FN {m['fn']}]", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=ink)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in metrics_order], color=ink2, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=muted, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title("Random forest vs rule-based detector", loc="left", fontsize=14, color=ink, pad=34)
    ax.text(0, 1.045, f"Scored on the same {len(y_va):,} validation queries " f"({int(y_va.sum()):,} tunneling, {int((1 - y_va).sum()):,} normal). " "Scores are for the tunneling class; FP/FN are counts.", transform=ax.transAxes, fontsize=9.5, color=ink2, va="bottom")

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=1, frameon=False, fontsize=9.5)

    for t in leg.get_texts():
        t.set_color(ink2)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=160, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.plot}")

if __name__ == "__main__":
    main()