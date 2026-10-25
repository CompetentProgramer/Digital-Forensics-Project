import pandas as pd

from Baseline.dns_tunnel_detector import extract_features

def load_data (file_path):
    data_set = pd.read_csv(file_path, header=None, names=["label", "domain"])
    data_set = data_set.dropna(subset=["label", "domain"])

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

    y = data_set["label"].values.astype(int)

    return x, y
