# DNS Tunneling Detection: Rules vs. Random Forest

Digital Forensics group project. We reproduce the rule-based and ML ideas from *DNS Tunneling: Threat Landscape and Improved Detection Solutions* (Işık et al., arXiv:2507.10267) and compare a rule-based pcap detector against a random forest, trained on one dataset and tested on another.

## Layout

| Folder | Contents |
|---|---|
| `Baseline/` | Rule-based detector and the comparison scripts |
| `ML/` | Random forest training pipeline and the Bubnov training/validation data |
| `Metrics/` | Result CSVs and charts written by the scripts |
| `data/` | Palau et al. (2019) DNS Threats Dataset, used only as a test set |

### Baseline/
| File | Purpose |
|---|---|
| `dns_tunnel_detector.py` | Reads a pcap, computes features, applies six rules, prints a summary, writes a CSV |
| `make_test_pcap.py` | Writes a synthetic `test_dns.pcap` (120 normal, 80 tunnel-like queries) |
| `compare_rf_vs_rules.py` | Forest vs rules on the Bubnov validation file; writes to `Metrics/` |
| `cross_dataset_test.py` | Trains on Bubnov, tests on Palau; writes to `Metrics/` |
| `make_augmented_training.py` | Adds normal Palau domains to the training file (`ML/dataset/training_augmented.csv`) |

### ML/
`main.py` trains (or loads) the random forest and prints accuracy and a confusion matrix. `data_loader.py` and `train.py` are shared with the Baseline scripts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install scapy scikit-learn matplotlib numpy tldextract pandas joblib
```

## Run (from the repo root)

```bash
# Rule-based detector on a test pcap (expect: 200 parsed, 80 flagged)
python Baseline/make_test_pcap.py
python Baseline/dns_tunnel_detector.py test_dns.pcap --summary

# Random forest on Bubnov (delete ML/saved_models/*.joblib first if the data or features changed)
python ML/main.py

# Forest vs rules, same validation data
python Baseline/compare_rf_vs_rules.py

# Train on Bubnov, test on Palau
python Baseline/cross_dataset_test.py --palau data/test_combined_multiclass.csv.gz

# After retraining on the augmented file
python Baseline/make_augmented_training.py --palau data/test_combined_multiclass.csv.gz --n-normal 5000
python Baseline/cross_dataset_test.py --palau data/test_combined_multiclass.csv.gz \
    --train ML/dataset/training_augmented.csv --out Metrics/cross_dataset_metrics_augmented.csv
```

Only analyze captures you are authorized to use (course lab data, public datasets, or your own traffic).

## How the rule-based detector works

`extract_features()` computes length, Shannon entropy, subdomain length and entropy, label count, longest label, and digit ratio. `apply_rules()` flags a query when any rule fires:

| Rule | Fires when |
|---|---|
| `long_name` | name length > 50 |
| `high_subdomain_entropy` | subdomain entropy > 3.5 and subdomain at least 10 characters |
| `long_label` | any label > 30 characters |
| `many_labels` | more than 6 labels |
| `suspicious_qtype` | record type is TXT or NULL |
| `many_unique_subdomains` | one host sends at least 50 unique subdomains under one base domain |

Thresholds are judgment calls and can be changed with command-line flags (`--length`, `--entropy`, `--label`, `--labels`, `--rate`). Subdomain splitting uses the public suffix list through `tldextract`.

## Results

Fill in from `Metrics/` after the final runs.

| Setup | Precision | Recall | False alarms |
|---|---|---|---|
| Bubnov validation: random forest | 1.000 | 1.000 | 0 |
| Bubnov validation: rules (default) | 1.000 | 0.748 | 0 |
| Palau: random forest (trained on Bubnov) | 0.217 | 0.999 | 5,611 |
| Palau: rules (default) | 0.563 | 0.591 | 715 |
| Palau: random forest (augmented training) | | | |

## Limitations

- Bubnov's classes differ mainly in whether a subdomain exists, so a near-perfect score there is not strong evidence. The forest's false alarms on Palau show this.
- Palau's tunneling class is only 1,559 queries from 2 base domains.
- Rule thresholds are heuristics; the length cutoff tuned on Bubnov does not carry over to Palau.
- The pcap detector was tested on a synthetic capture only. It is not a Wireshark plugin.