import pandas as pd

from Baseline.dns_tunnel_detector import extract_features


def load_data (file_path):
    data_set = pd.read_csv(file_path)

    x = []
    for domain in data_set:
        x.append(extract_features(domain))

    y = data_set.values

    return x, y
