import itertools
import subprocess
import os
import pandas as pd
import time
import re
import sys

# ==========================================================
# Hyperparameter Space
# ==========================================================

seq_lens = [6, 8, 10]
latent_dims = [16, 32]
hidden_dims = [64, 128]
lrs = [1e-3]
percentiles = [95, 97, 99]

search_space = list(itertools.product(
    seq_lens,
    latent_dims,
    hidden_dims,
    lrs,
    percentiles
))

print(f"Total configurations: {len(search_space)}")

os.makedirs("models", exist_ok=True)

results = []

# ==========================================================
# Helper to Extract Metrics
# ==========================================================

def extract_metrics(output):
    acc = float(re.search(r"Accuracy\s*:\s*([0-9.]+)", output).group(1))
    prec = float(re.search(r"Precision\s*:\s*([0-9.]+)", output).group(1))
    rec = float(re.search(r"Recall\s*:\s*([0-9.]+)", output).group(1))
    f1 = float(re.search(r"F1 Score\s*:\s*([0-9.]+)", output).group(1))
    return acc, prec, rec, f1

# ==========================================================
# Grid Search
# ==========================================================

for i, (seq_len, latent_dim, hidden_dim, lr, percentile) in enumerate(search_space):

    print(f"\n[{i+1}/{len(search_space)}] Running:")
    print(f"seq={seq_len}, latent={latent_dim}, hidden={hidden_dim}, "
          f"lr={lr}, percentile={percentile}")

    # -----------------------------
    # 1️⃣ Train
    # -----------------------------
    train_cmd = [
        sys.executable,
        "-m",
        "src.ai.autoencoder.train",
        "--seq_len", str(seq_len),
        "--latent_dim", str(latent_dim),
        "--hidden_dim", str(hidden_dim),
        "--lr", str(lr),
        "--percentile", str(percentile),
        "--architecture", "kitsune"
    ]

    subprocess.run(train_cmd, check=True)

    model_path = (
        f"models/AE_seq{seq_len}"
        f"_lat{latent_dim}"
        f"_hid{hidden_dim}"
        f"_lr{lr}"
        f"_p{percentile}.pt"
    )

    # -----------------------------
    # 2️⃣ Evaluate BENIGN
    # -----------------------------
    benign_output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "src.ai.autoencoder.test",
            "--model_path", model_path,
            "--label", "benign"
        ],
        text=True
    )

    benign_acc, benign_prec, benign_rec, benign_f1 = extract_metrics(benign_output)

    # -----------------------------
    # 3️⃣ Evaluate ABNORMAL
    # -----------------------------
    abnormal_output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "src.ai.autoencoder.test",
            "--model_path", model_path,
            "--label", "abnormal"
        ],
        text=True
    )

    abnormal_acc, abnormal_prec, abnormal_rec, abnormal_f1 = extract_metrics(abnormal_output)

    # -----------------------------
    # Store Results
    # -----------------------------
    results.append({
        "seq_len": seq_len,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "percentile": percentile,

        "benign_accuracy": benign_acc,
        "benign_precision": benign_prec,
        "benign_recall": benign_rec,
        "benign_f1": benign_f1,

        "abnormal_accuracy": abnormal_acc,
        "abnormal_precision": abnormal_prec,
        "abnormal_recall": abnormal_rec,
        "abnormal_f1": abnormal_f1
    })

    time.sleep(1)

# ==========================================================
# Ranking (by abnormal F1)
# ==========================================================

df = pd.DataFrame(results)

df = df.sort_values("abnormal_f1", ascending=False)

df.to_csv("models/hpo_results_ranked.csv", index=False)

print("\nTop Configurations:")
print(df.head())