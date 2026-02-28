import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import Autoencoder
from .encoder import Encoder


# ==========================================================
# Argument Parser
# ==========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--seq_len", type=int, default=6)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=16)
parser.add_argument("--architecture", type=str, default="kitsune")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--percentile", type=int, default=99)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


BASE_DIR = os.path.dirname(__file__)
data_folder = os.path.join(BASE_DIR, "5g-mobiwatch")

csv_path = os.path.join(
    data_folder,
    "5g-mobiwatch_benign_mobiflow.csv"
)

print("Loading dataset from:", csv_path)

df = pd.read_csv(csv_path, delimiter=";")
df.fillna(0, inplace=True)


encoder = Encoder()
X_sequences = encoder.encode_mobiflow(df, args.seq_len)

print("Encoded shape:", X_sequences.shape)


X_train, X_val = train_test_split(
    X_sequences,
    test_size=0.2,
    random_state=args.seed
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, X_train),
    batch_size=args.batch_size,
    shuffle=True
)


input_dim = X_train.shape[1]

model = Autoencoder(
    input_dim=input_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    architecture=args.architecture
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


best_val_loss = float("inf")
patience = 20
counter = 0

os.makedirs("models", exist_ok=True)

print("\nStarting training...\n")

for epoch in range(args.epochs):

    model.train()
    train_loss = 0

    for batch in train_loader:
        x, _ = batch
        output = model(x)
        loss = criterion(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, X_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "models/best_val_loss.pt")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train {train_loss:.6f} | Val {val_loss:.6f}")


model.load_state_dict(torch.load("models/best_val_loss.pt", weights_only=False))
model.eval()


with torch.no_grad():
    recon = model(X_train)
    train_error = torch.mean((X_train - recon) ** 2, dim=1)

threshold = np.percentile(train_error.cpu().numpy(), args.percentile)

print("Threshold selected:", threshold)


model_path = (
    f"models/AE_seq{args.seq_len}"
    f"_lat{args.latent_dim}"
    f"_hid{args.hidden_dim}"
    f"_lr{args.lr}"
    f"_p{args.percentile}.pt"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "threshold": threshold,
    "config": vars(args)
}, model_path)

print("Model saved to:", model_path)
print("Training complete.\n")