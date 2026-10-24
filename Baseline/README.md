# DNS Tunneling Baseline Detector

A rule-based detector that reads a pcap file, pulls out the DNS queries, and flags ones that look like DNS tunneling. It is the rule-based baseline for our reproduction of *DNS Tunneling: Threat Landscape and Improved Detection Solutions* (Işık et al., arXiv:2507.10267), which classifies domains by **query length** and **Shannon entropy**.

## Files

| File | Purpose |
|---|---|
| `make_test_pcap.py` | Writes a synthetic `test_dns.pcap` (normal lookups plus tunnel-shaped queries). Nothing is sent on the network. |
| `dns_tunnel_detector.py` | Reads a pcap, computes features, applies rules, prints a summary, and writes a CSV. |

## Run it

```bash
pip install scapy

python make_test_pcap.py
python dns_tunnel_detector.py test_dns.pcap --summary --csv results.csv
```

Expected output on the test file: `DNS queries parsed: 200` and `Flagged suspicious: 80 (40.0%)`.

Optional thresholds (defaults shown):

```bash
python dns_tunnel_detector.py capture.pcap --entropy 3.5 --length 50 --label 30 --labels 6 --rate 50
```

Only analyze captures you are authorized to use (course lab data, public datasets, or your own traffic).

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
- `split_domain()` assumes the last two labels are the registered domain, so `example.co.uk` is split incorrectly. Use `tldextract` if this matters.
- The test pcap is synthetic and easy. Real evaluation needs the public datasets used in the paper.
- This runs offline on pcap files. It is not a Wireshark plugin.
