#!/usr/bin/env python3
"""Write a SYNTHETIC test pcap.
Mix of normal-looking lookups and random base32-style subdomains that mimic
the shape of tunneling queries, for validating the detector."""
import base64
import os
import random

from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

random.seed(7)
CLIENT_NORMAL, CLIENT_TUNNEL, RESOLVER = "10.0.0.5", "10.0.0.99", "10.0.0.2"
packets, t = [], 1_700_000_000.0


def q(src, name, qtype="A"):
    global t
    t += random.uniform(0.01, 0.5)
    p = (Ether() / IP(src=src, dst=RESOLVER) / UDP(sport=random.randint(20000, 60000), dport=53)
         / DNS(rd=1, qd=DNSQR(qname=name, qtype=qtype)))
    p.time = t
    packets.append(p)


normal = ["www.google.com", "mail.example.org", "cdn.cloudflare.com", "login.microsoft.com",
          "api.github.com", "en.wikipedia.org", "static.fpu.edu", "update.ubuntu.com"]
for _ in range(120):
    q(CLIENT_NORMAL, random.choice(normal))

for _ in range(80):  # tunnel-shaped: encoded payload in subdomain labels, TXT queries
    payload = base64.b32encode(os.urandom(random.randint(25, 35))).decode().lower().rstrip("=")
    q(CLIENT_TUNNEL, f"{payload[:30]}.{payload[30:]}.t.lab-example.net", "TXT")

wrpcap("test_dns.pcap", packets)
print(f"wrote {len(packets)} packets")