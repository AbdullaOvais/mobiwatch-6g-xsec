import argparse
import pandas as pd
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .model import Autoencoder
from .encoder import Encoder


def save_anomalies(df, error, threshold, seq_len):
    anomaly_indices = (error > threshold).nonzero(as_tuple=True)[0]

    rows = []

    for idx in anomaly_indices:

        start = int(idx)
        end = start + seq_len

        # Safety check to avoid dataframe overflow
        if end > len(df):
            continue

        sequence_msgs = df.iloc[start:end]["msg"].tolist()

        rows.append({
            "sequence_id": int(idx),
            "reconstruction_error": float(error[idx]),
            "sequence_length": seq_len,
            "first_message": sequence_msgs[0],
            "last_message": sequence_msgs[-1],
            "message_sequence": " -> ".join(sequence_msgs)
        })

    out = pd.DataFrame(rows)

    out.to_csv("detected_anomalies.csv", index=False)

    print(f"\nSaved {len(rows)} anomalous sequences to detected_anomalies.csv")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--label", type=str, default="combined")

    args = parser.parse_args()

    print("\nLoading trained model...\n")

    checkpoint = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=False
    )

    config = checkpoint["config"]
    threshold = checkpoint["threshold"]
    state_dict = checkpoint["model_state_dict"]

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

        return torch.tensor(X, dtype=torch.float32), df

    # --------------------------------------------------
    # Dataset loading
    # --------------------------------------------------

    if args.label == "combined":

        print("Loading BENIGN and ABNORMAL datasets...\n")

        X_benign, df_benign = load_dataset("benign")
        X_abnormal, df_abnormal = load_dataset("abnormal")

        X_test = torch.cat([X_benign, X_abnormal], dim=0)

        df_combined = pd.concat(
            [df_benign, df_abnormal],
            ignore_index=True
        )

        y_true = np.concatenate([
            np.zeros(len(X_benign)),
            np.ones(len(X_abnormal))
        ])

    else:

        print(f"Loading {args.label.upper()} dataset...\n")

        X_test, df_combined = load_dataset(args.label)

        y_true = np.ones(len(X_test)) if args.label == "abnormal" else np.zeros(len(X_test))

    # --------------------------------------------------
    # Model reconstruction
    # --------------------------------------------------

    model = Autoencoder(
        input_dim=X_test.shape[1],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        architecture=config["architecture"]
    )

    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        recon = model(X_test)
        error = torch.mean((X_test - recon) ** 2, dim=1)

    y_pred = (error > threshold).int().numpy()

    # --------------------------------------------------
    # Save anomalies for LLM explainability
    # --------------------------------------------------

    save_anomalies(
        df_combined,
        error,
        threshold,
        config["seq_len"]
    )

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===== METRICS =====")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nTesting complete.\n")


if __name__ == "__main__":
    main()