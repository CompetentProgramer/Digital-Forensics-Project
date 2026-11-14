import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Baseline"))
from dns_tunnel_detector import extract_features  # noqa: E402

FEATURE_NAMES = ["sub_length", "sub_entropy", "num_labels", "max_label_len", "digit_ratio"]


def features_from_domains(domains):
    """Feature vectors for a list of domain names (also used by Baseline/ scripts)."""
    x = []
    for domain in domains:
        feat_dict = extract_features(domain, "A")
        x.append([feat_dict[name] for name in FEATURE_NAMES])
    return x


def load_data (file_path):
    data_set = pd.read_csv(file_path, header=None, names=["label", "domain"])
    data_set = data_set.dropna(subset=["label", "domain"])

    x = features_from_domains(data_set["domain"])
    y = data_set["label"].values.astype(int)

    return x, y