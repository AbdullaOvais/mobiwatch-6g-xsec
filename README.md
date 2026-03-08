# MobiWatch Reproduction Study

This repository contains a **reproduction study of the anomaly detection models used in the MobiWatch framework** proposed in the paper:

**6G-XSec: Explainable Edge Security for Emerging OpenRAN Architectures (HotNets 2024).**

The goal of this work is to **reproduce and evaluate the unsupervised anomaly detection models** used in the MobiWatch xApp for detecting signaling-level attacks in 5G networks.

The models evaluated in this reproduction include:

- **Autoencoder (Reconstruction-based anomaly detection)**
- **LSTM (Sequence prediction–based anomaly detection)**

Both models are trained **only on benign cellular traffic** and evaluated on benign and attack datasets.

---

# Original Project

Original MobiWatch repository:  
https://github.com/5GSEC/MobiWatch

Original research paper:  
https://onehouwong.github.io/papers/HotNets_2024_6gxsec.pdf

The original system integrates:

- MobiFlow telemetry
- Deep learning anomaly detection
- Explainability using LLMs
- Deployment as an O-RAN xApp

This reproduction study focuses **only on the anomaly detection models**.

---

# Project Overview

The reproduced pipeline follows the same design as the original system:

```
PCAP Traffic
      ↓
MobiFlow Telemetry (CSV)
      ↓
Sequence Encoding
      ↓
Deep Learning Models
   ├── Autoencoder
   └── LSTM
      ↓
Anomaly Score
      ↓
Percentile Threshold Detection
```

The system detects cellular signaling anomalies such as:

- Blind DoS
- BTS DoS
- IMSI extraction
- Identity leakage
- Null ciphering attacks

---

# Dataset

The dataset used for reproduction is included in the repository.

Two formats are provided:

### PCAP Files

Original network captures collected from a 5G testbed.

```
dataset/pcap/5g-benign/
dataset/pcap/5g-attack/
```

### MobiFlow Telemetry

Converted security telemetry used for model training.

```
dataset/mobiflow/
    5g-mobiwatch_benign_mobiflow.csv
    5g-mobiwatch_abnormal_mobiflow.csv
```

Telemetry features include:

- RRC signaling messages
- NAS signaling messages
- RNTI identifiers
- TMSI identifiers
- Ciphering algorithms
- Integrity protection information

Only **benign traffic is used for training**.

---

# Environment Setup

Create a Python environment and install dependencies.

```
python -m venv mobiwatch_env
source mobiwatch_env/bin/activate
```

Install required packages:

```
pip install torch
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

---

# Autoencoder Model

The Autoencoder model learns a compressed representation of normal cellular traffic.

During inference:

```
Reconstruction Error = MSE(input − reconstructed)
```

If the error exceeds a threshold, the sequence is classified as **anomalous**.

### Train Autoencoder

```
python -m src.ai.autoencoder.train
```

### Test Autoencoder (Benign)

```
python -m src.ai.autoencoder.test \
--model_path models/AE_model.pt \
--label benign
```

### Test Autoencoder (Abnormal)

```
python -m src.ai.autoencoder.test \
--model_path models/AE_model.pt \
--label abnormal
```

---

# LSTM Model

The LSTM model performs **next-step prediction on telemetry sequences**.

During inference:

```
Prediction Error = RMSE(predicted − actual)
```

If prediction error exceeds the threshold, the sequence is classified as **anomalous**.

### Train LSTM

```
python -m src.ai.lstm.train
```

### Test LSTM

```
python -m src.ai.lstm.test
```

---

# Hyperparameter Optimization

Grid search was performed over the following parameters:

| Parameter | Values Tested |
|--------|--------|
| Sequence Length | 6, 8, 10 |
| Latent Dimension | 16, 32 |
| Hidden Dimension | 64, 128 |
| Threshold Percentile | 95, 97, 99 |
| Learning Rate | 0.001 |

Best Autoencoder configuration:

```
Sequence length = 10
Latent dimension = 16
Hidden dimension = 64
Threshold percentile = 95
```

---

# Reproduced Results

## Autoencoder

| Metric | Benign | Abnormal |
|------|------|------|
| Accuracy | 93.10% | 92.48% |
| Precision | 93.10% | 100% |
| Recall | — | 92.48% |
| F1 Score | — | 96.09% |

---

## LSTM

| Metric | Benign | Abnormal |
|------|------|------|
| Accuracy | 89.59% | 88.67% |
| Precision | 89.59% | 100% |
| Recall | — | 88.67% |
| F1 Score | — | 94.51% |

---

# Observations

Key findings from the reproduction study:

- Autoencoder achieved **higher recall and F1-score** than LSTM.
- Both models successfully detected previously unseen attacks.
- Detection performance is sensitive to **threshold percentile selection**.
- Longer sequence windows improved anomaly detection performance.

Although the exact metrics differ slightly from the original paper, the **overall detection trends remain consistent**.

---

# Project Structure

```
MobiWatch-main/
│
├── dataset/
│   ├── mobiflow/
│   └── pcap/
│
├── src/
│   ├── ai/
│   │   ├── autoencoder/
│   │   └── lstm/
│
├── models/
│
├── hpo.py
├── train.py
└── test.py
```

---

# Citation

If you use this work, please cite the original paper:

```
@inproceedings{6G-XSEC:Hotnets24,
  title     = {6G-XSec: Explainable Edge Security for Emerging OpenRAN Architectures},
  author    = {Wen, Haohuang and Sharma, Prakhar and Yegneswaran, Vinod and Porras, Phillip and Gehani, Ashish and Lin, Zhiqiang},
  booktitle = {Proceedings of the Twenty-Third ACM Workshop on Hot Topics in Networks (HotNets 2024)},
  year      = 2024
}
```
