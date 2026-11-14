# DNS Tunneling Baseline Detector

A rule-based detector that reads a pcap file, pulls out the DNS queries, and flags ones that look like DNS tunneling. It is the rule-based baseline for our reproduction of *DNS Tunneling: Threat Landscape and Improved Detection Solutions* (Işık et al., arXiv:2507.10267), which classifies domains by **query length** and **Shannon entropy**.

## Files

| File | Purpose |
|---|---|
| `make_test_pcap.py` | Writes a synthetic `test_dns.pcap` (normal lookups plus tunnel-shaped queries). Nothing is sent on the network. |
| `dns_tunnel_detector.py` | Reads a pcap, computes features, applies rules, prints a summary, and writes a CSV. |
| `compare_rf_vs_rules.py` | Scores the teammate's random forest (`ML/`) and the rule detector on the same validation file; writes `comparison_metrics.csv` and `comparison.png`. Run `python compare_rf_vs_rules.py` from `Baseline/`. |
| `cross_dataset_test.py` | Trains the random forest on Bubnov and scores it and the rule detector on the Palau DNS Threats test file (accepts `.csv.gz`). Run `python cross_dataset_test.py --palau path/to/file.csv.gz` from `Baseline/`; needs `compare_rf_vs_rules.py` in the same folder. |
| `ml_reproduction.py` | Trains the paper's models on a labeled domain CSV with a clean train/test split, compares them with the rule baseline, writes `metrics.csv` and `decision_boundary.png`. |

## Run it

```bash
pip install scapy scikit-learn matplotlib numpy tldextract

python make_test_pcap.py
python dns_tunnel_detector.py test_dns.pcap --summary --csv results.csv
```

Expected output on the test file: `DNS queries parsed: 200` and `Flagged suspicious: 80 (40.0%)`.

Optional thresholds (defaults shown):

```bash
python dns_tunnel_detector.py capture.pcap --entropy 3.5 --length 50 --label 30 --labels 6 --rate 50
```

Only analyze captures you are authorized to use (course lab data, public datasets, or your own traffic).

## Run the ML comparison

```bash
# 1. Pipeline test with built-in SYNTHETIC data (results are NOT evidence)
python ml_reproduction.py --synthetic

# 2. Real run: a CSV with a domain column and a label column
python ml_reproduction.py --data dataset.csv --domain-col domain --label-col label
```

Get the public datasets from the paper's references (Palau et al. and the Bubnov dataset on Mendeley), open the CSV, and pass its real column names with `--domain-col` and `--label-col`. Labels of `1`, `tunnel`, `tunneling`, `malicious` or `attack` count as tunneling; everything else counts as normal.

What it does:

- Computes the paper's two features, domain length and Shannon entropy.
- Splits the data once into 80% train and 20% test (stratified, fixed seed). The test set is never used for fitting or scaling.
- Trains Multinomial NB, Bernoulli NB, Linear SVM, and a 2-64-32-1 neural network (scikit-learn `MLPClassifier`, same shape as the paper's PyTorch model).
- Scores the rule baseline from `dns_tunnel_detector.py` on the same split, using only the rules that work on a bare domain (long name, high subdomain entropy, long label, many labels).
- Prints train and test scores side by side, which tests whether the paper's Fig. 7 reports training-set results.
- Writes `metrics.csv` and a decision boundary plot of the neural net on the test points.

Notes: Naive Bayes needs non-negative inputs, so Multinomial NB uses the raw features and Bernoulli NB binarizes at the training median (its default threshold of 0 makes every feature look "on"). Precision, recall and F1 are for the tunneling class.

## How the detector works

1. **Read packets.** `read_queries()` streams the pcap with scapy and keeps only DNS *queries* (not responses), returning the time, source IP, query name, and record type.
2. **Compute features.** `extract_features()` calculates, per query:
   - `length`: characters in the full name
   - `entropy`: Shannon entropy of the name (the randomness of its characters, as in the paper)
   - `sub_length` and `sub_entropy`: the same for the subdomain part only
   - `num_labels`, `max_label_len`, `digit_ratio`
3. **Apply rules.** `apply_rules()` checks simple, explainable rules and records which fired:

   | Rule | Fires when |
   |---|---|
   | `long_name` | name length > `--length` (50) |
   | `high_subdomain_entropy` | subdomain entropy > `--entropy` (3.5) and subdomain is at least 10 characters |
   | `long_label` | any single label > `--label` (30) |
   | `many_labels` | more than `--labels` (6) labels |
   | `suspicious_qtype` | record type is TXT or NULL |
   | `many_unique_subdomains` | one host sends at least `--rate` (50) unique subdomains under one base domain |

4. **Report.** Any query with at least one rule hit is `suspicious`. The script prints totals, the top flagged domains (`--summary`), sample hits, and writes every query to a CSV (`--csv`).

Why entropy works: normal names like `www.google.com` are made of readable words, so their characters repeat in predictable ways. Tunneling tools encode data into random-looking strings, which pushes entropy up and makes names long.

## Limitations

- The rules and thresholds are heuristics. They can produce false positives (for example CDN or tracking domains with long random names) and miss slow, low-volume tunnels.
- `split_domain()` uses the public suffix list through `tldextract`, so `example.co.uk` is split correctly. Without `tldextract` installed it falls back to a built-in list of common two-part suffixes, which is incomplete.
- The test pcap and the `--synthetic` dataset are generated. They show the code runs; real evaluation needs the public datasets used in the paper.
- The ML comparison uses only domain length and entropy, like the paper, so it cannot see packet context (record type, query rate) that the pcap rules use.
- This runs offline on pcap files. It is not a Wireshark plugin.