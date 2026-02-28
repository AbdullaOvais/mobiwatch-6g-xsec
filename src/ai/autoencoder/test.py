import argparse
import pandas as pd
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .model import Autoencoder
from .encoder import Encoder


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--label", type=str, default="combined")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, weights_only=False)
    config = checkpoint["config"]
    threshold = checkpoint["threshold"]

    BASE_DIR = os.path.dirname(__file__)
    data_folder = os.path.join(BASE_DIR, "5g-mobiwatch")

    encoder = Encoder()

    def load_dataset(label):
        csv_path = os.path.join(
            data_folder,
            f"5g-mobiwatch_{label}_mobiflow.csv"
        )
        df = pd.read_csv(csv_path, delimiter=";")
        df.fillna(0, inplace=True)
        X = encoder.encode_mobiflow(df, config["seq_len"])
        return torch.tensor(X, dtype=torch.float32)

    if args.label == "combined":
        X_benign = load_dataset("benign")
        X_abnormal = load_dataset("abnormal")

        X_test = torch.cat([X_benign, X_abnormal], dim=0)
        y_true = np.concatenate([
            np.zeros(len(X_benign)),
            np.ones(len(X_abnormal))
        ])
    else:
        X_test = load_dataset(args.label)
        y_true = np.ones(len(X_test)) if args.label == "abnormal" else np.zeros(len(X_test))

    model = Autoencoder(
        input_dim=X_test.shape[1],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        architecture=config["architecture"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        recon = model(X_test)
        error = torch.mean((X_test - recon) ** 2, dim=1)

    y_pred = (error > threshold).int().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===== METRICS =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


if __name__ == "__main__":
    main()