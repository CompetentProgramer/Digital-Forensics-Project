#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict

from scapy.all import DNS, DNSQR, IP, IPv6, PcapReader, UDP, TCP  # noqa: F401

QTYPES = {1: "A", 2: "NS", 5: "CNAME", 10: "NULL", 15: "MX", 16: "TXT", 28: "AAAA", 255: "ANY"}
SUSPICIOUS_QTYPES = {"TXT", "NULL"}  # commonly used by tunneling tools


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits/char (same idea as the paper's Listing 2,
    but iterating only over characters that occur)."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def split_domain(qname: str):
    """Return (subdomain, registered_domain) using a simple last-two-labels
    heuristic. Swap in `tldextract` for production."""
    labels = qname.rstrip(".").lower().split(".")
    if len(labels) <= 2:
        return "", ".".join(labels)
    return ".".join(labels[:-2]), ".".join(labels[-2:])


def extract_features(qname: str, qtype: str) -> dict:
    sub, base = split_domain(qname)
    name = qname.rstrip(".").lower()
    labels = name.split(".")
    return {
        "qname": name,
        "qtype": qtype,
        "length": len(name),
        "entropy": round(shannon_entropy(name), 3),
        "subdomain": sub,
        "base_domain": base,
        "sub_length": len(sub),
        "sub_entropy": round(shannon_entropy(sub.replace(".", "")), 3),
        "num_labels": len(labels),
        "max_label_len": max(len(l) for l in labels),
        "digit_ratio": round(sum(ch.isdigit() for ch in name) / max(len(name), 1), 3),
    }


def apply_rules(f: dict, args) -> list:
    """Each rule is simple and explainable. Returns list of triggered rule names."""
    hits = []
    if f["length"] > args.length:
        hits.append("long_name")
    if f["sub_entropy"] > args.entropy and f["sub_length"] >= 10:
        hits.append("high_subdomain_entropy")
    if f["max_label_len"] > args.label:
        hits.append("long_label")
    if f["num_labels"] > args.labels:
        hits.append("many_labels")
    if f["qtype"] in SUSPICIOUS_QTYPES:
        hits.append("suspicious_qtype")
    return hits


def read_queries(pcap_path: str):
    """Yield (timestamp, src_ip, qname, qtype) for each DNS query packet."""
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if not (pkt.haslayer(DNS) and pkt.haslayer(DNSQR)):
                continue
            dns = pkt[DNS]
            if dns.qr != 0:  # 0 = query, 1 = response
                continue
            src = pkt[IP].src if pkt.haslayer(IP) else (pkt[IPv6].src if pkt.haslayer(IPv6) else "?")
            raw = dns.qd.qname
            qname = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            qtype = QTYPES.get(dns.qd.qtype, str(dns.qd.qtype))
            yield float(pkt.time), src, qname, qtype


def main():
    ap = argparse.ArgumentParser(description="Rule-based DNS tunneling detector for PCAPs")
    ap.add_argument("pcap")
    ap.add_argument("--csv", help="write per-query results to this CSV")
    ap.add_argument("--summary", action="store_true", help="print per-base-domain summary")
    ap.add_argument("--entropy", type=float, default=3.5, help="subdomain entropy threshold (default 3.5)")
    ap.add_argument("--length", type=int, default=50, help="total name length threshold (default 50)")
    ap.add_argument("--label", type=int, default=30, help="max single-label length threshold (default 30)")
    ap.add_argument("--labels", type=int, default=6, help="label count threshold (default 6)")
    ap.add_argument("--rate", type=int, default=50,
                    help="unique subdomains per base domain per source to flag (default 50)")
    args = ap.parse_args()

    rows = []
    unique_subs = defaultdict(set)  # (src, base) -> set of subdomains

    try:
        for ts, src, qname, qtype in read_queries(args.pcap):
            f = extract_features(qname, qtype)
            f["timestamp"] = ts
            f["src"] = src
            f["rules"] = apply_rules(f, args)
            if f["subdomain"]:
                unique_subs[(src, f["base_domain"])].add(f["subdomain"])
            rows.append(f)
    except FileNotFoundError:
        sys.exit(f"File not found: {args.pcap}")

    # Rate rule: many unique subdomains under one base domain from one host
    flagged_pairs = {k for k, v in unique_subs.items() if len(v) >= args.rate}
    for f in rows:
        if (f["src"], f["base_domain"]) in flagged_pairs:
            f["rules"].append("many_unique_subdomains")
        f["suspicious"] = bool(f["rules"])
        f["rules"] = ";".join(f["rules"])

    total = len(rows)
    sus = [r for r in rows if r["suspicious"]]
    print(f"DNS queries parsed: {total}")
    print(f"Flagged suspicious: {len(sus)} ({(100 * len(sus) / total) if total else 0:.1f}%)")

    if args.summary and sus:
        print("\nTop flagged base domains:")
        counts = Counter((r["src"], r["base_domain"]) for r in sus)
        for (src, base), n in counts.most_common(10):
            print(f"  {src:<16} {base:<35} {n:>5} flagged queries")

    print("\nSample flagged queries:")
    for r in sus[:10]:
        print(f"  [{r['rules']}] len={r['length']} H={r['entropy']} {r['qname'][:70]}")

    if args.csv:
        fields = ["timestamp", "src", "qname", "qtype", "length", "entropy", "subdomain",
                  "base_domain", "sub_length", "sub_entropy", "num_labels", "max_label_len",
                  "digit_ratio", "suspicious", "rules"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()