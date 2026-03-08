import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from lstm_multivariate import train, test, test_from_iter
from utils import validate_by_rmse, Normalizer
from encoder import Encoder
from torch.utils.data import DataLoader, TensorDataset
import sys
import more_itertools
import pickle

# training dataset
train_dataset = "5g-mobiwatch"
train_label = "benign"
delimeter = ";"

# Step 1: Load and preprocess data
data_folder = "../../../dataset/mobiflow"
df = pd.read_csv(f'{data_folder}/{train_dataset}_{train_label}_mobiflow.csv', header=0, delimiter=delimeter)
# Handle missing values
df.fillna(0, inplace=True)

sequence_length = 10 # use X to predict next
encoder = Encoder()
X_sequences = encoder.encode_mobiflow(df, sequence_length+1)
print(X_sequences.shape)

x_train = []
y_train = []
for i in range(len(X_sequences)):
    split_seq = np.split(X_sequences[i], sequence_length+1)
    x_train.append(split_seq[:sequence_length])
    y_train.append(split_seq[-1])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape, y_train.shape)

# ======================
# Normalization
# ======================
normer = Normalizer(x_train.shape[-1], online_minmax=True)

x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
x_train_reshaped = normer.fit_transform(x_train_reshaped)
x_train = x_train_reshaped.reshape(x_train.shape)

y_train = normer.transform(y_train)


pickle.dump(normer, open("save/normer.pkl", "wb"))

seed = 42
val_portion = 0.1 # size of validation set
indices = np.arange(x_train.shape[0])
x_train, x_val, indices_train, indices_val = train_test_split(x_train, indices, test_size=val_portion, random_state=seed)
y_val = y_train[indices_val]
y_train = y_train[indices_train]
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# Split data into training and test sets
# seed = 30 # 42
# indices = np.arange(X_sequences.shape[0])
# X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=0.2, random_state=seed)
# X_train, X_test, indices_train, indices_test = train_test_split(X_sequences, indices, test_size=0.01, random_state=seed)


# Convert to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)

# # Create DataLoader for training
# train_dataset = TensorDataset(X_train, X_train)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# x_train = more_itertools.windowed(X_sequences,n=sequence_length,step=1)
# x_train = np.asarray(list(x_train)[:-1])
# y_train = np.asarray(X_sequences[sequence_length:])
# print(x_train.shape)
# print(y_train.shape)

# id_column_idx = [1, 2]
# for x in X_train:
#     for idx in id_column_idx:
#         # handle TMSI = -1?
#         if idx == 2:
#             is_tmsi = True
#         else:
#             is_tmsi = False
#         encoded_vals = encode_value_within_window(x[:, idx], is_tmsi)
#         x[:, idx] = encoded_vals

model, thres, rmse_vec = train(x_train, y_train)
torch.save({'net':model,'thres':thres},f'./save/lstm_multivariate_{train_dataset}_{train_label}.pth.tar')    

# validation
rmse_vec = test(model, thres, x_val, y_val)
anomalies = (rmse_vec > thres).detach().cpu()
if len(anomalies) > 0:
    for anomalies_idx in torch.nonzero(anomalies).squeeze():
        df_idx = anomalies_idx.item()
        sequence_data = df.loc[df_idx:df_idx + sequence_length]
        df_sequence = pd.DataFrame(sequence_data, columns=encoder.identifier_features + encoder.numerical_features + encoder.categorical_features)
        print(df_sequence)
        print()
    
# ==========================
# Proper Window-Level Metrics
# ==========================

# pred_labels = anomalies.detach().cpu().numpy().astype(int)
# true_labels = np.zeros(len(pred_labels))

# for attack in gt.values():
#     for idx in attack:
#         if idx < len(true_labels):
#             true_labels[idx] = 1

# TP = ((pred_labels == 1) & (true_labels == 1)).sum()
# TN = ((pred_labels == 0) & (true_labels == 0)).sum()
# FP = ((pred_labels == 1) & (true_labels == 0)).sum()
# FN = ((pred_labels == 0) & (true_labels == 1)).sum()

# acc = 100 * (TP + TN) / (TP + TN + FP + FN)
# precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
# f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
# fpr = 100 * FP / (FP + TN) if (FP + TN) > 0 else 0

# print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
# print(f"Accuracy: {acc:.3f}%")
# print(f"Precision: {precision:.3f}%")
# print(f"Recall: {recall:.3f}%")
# print(f"F1-score: {f1:.3f}%")
# print(f"False Positive Rate: {fpr:.3f}%")
# ==========================
# Validation Metrics (Benign Only)
# ==========================

anomalies = (rmse_vec > thres).detach().cpu()

FP = anomalies.sum().item()
TN = len(anomalies) - FP

fpr = 100 * FP / (FP + TN)

print(f"Validation False Positives: {FP}")
print(f"Validation True Negatives: {TN}")
print(f"Validation FPR: {fpr:.3f}%")

plot = True
if plot:
    import matplotlib.pyplot as plt

    # Add these two lines to pull the data off the GPU
    rmse_plot = rmse_vec.cpu().detach().numpy()
    thres_plot = thres.cpu().detach().item()

    # Creating a simple line chart
    plt.figure(figsize=(10, 5))

    # Update these lines to use the new CPU variables
    plt.plot(rmse_plot, marker='o', linestyle='-', color='b')
    plt.axhline(y=thres_plot, color='r', linestyle='-')

    plt.title(f'LSTM RMSE (Threshold: {thres_plot:.4f})')
    plt.xlabel('Seq Index')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig("validation.png")

# normer = Normalizer(train_feat.shape[-1],online_minmax=True)
# train_feat = normer.fit_transform(train_feat)
# model, thres = train(train_feat, train_feat.shape[-1], batch_size, lr, weight_decay, epoches)
# save_data = {'net':model,'thres':thres, 'dataset':dataset_name, "batch_size":batch_size, "lr":lr, "weight_decay":weight_decay, "epoches":epoches}
# torch.save(save_data,'./save/autoencoder_%s.pth.tar' % (train_ver))

