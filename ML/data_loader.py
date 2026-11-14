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
    data_set = pd.read_csv(file_path, nrows=1)

    if "domain" in data_set.columns and "class" in data_set.columns:
        data_set = pd.read_csv(file_path)
        data_set = data_set.dropna(subset=["domain", "class"])

        y = data_set["class"]

    else:
        data_set = pd.read_csv(file_path, header=None, names=["label", "domain"])
        data_set = data_set.dropna(subset=["label", "domain"])

        #x = features_from_domains(data_set["domain"])
        y = data_set["label"].values.astype(int)

    x = []
    for domain in data_set["domain"]:
        feat_dict = extract_features(domain, "A")

        numeric_vector = [
            feat_dict["sub_length"],
            feat_dict["sub_entropy"],
            feat_dict["num_labels"],
            feat_dict["max_label_len"],
            feat_dict["digit_ratio"]
        ]

        x.append(numeric_vector)
        #print(f"[Data Loader] Ingested at {file_path}")

    return x, y