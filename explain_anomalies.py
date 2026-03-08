import pandas as pd
import ollama
import time

print("Loading anomalies...")
df = pd.read_csv("detected_anomalies.csv")

print(f"Total anomalies detected: {len(df)}\n")

for idx, row in df.iterrows():

    sequence_id = row["sequence_id"]
    error = row["reconstruction_error"]
    sequence = row["message_sequence"]

    prompt = f"""
You are an expert security analyst specializing in 5G and Open RAN networks.

A machine learning anomaly detection system flagged the following
5G signaling sequence as suspicious.

Sequence ID: {sequence_id}
Reconstruction Error: {error}

Sequence:
{sequence}

Typical normal signaling order:
RRCSetupRequest → RRCSetup → RRCSetupComplete →
RegistrationRequest → AuthenticationRequest →
AuthenticationResponse → SecurityModeCommand → SecurityModeComplete

Explain:

1. Why this sequence is anomalous
2. Which 5G protocol procedure is violated
3. Possible cellular attack causing this
4. Suggested mitigation
"""

    print("=================================================")
    print(f"Analyzing sequence_id: {sequence_id}")
    print("=================================================\n")

    stream = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    explanation = ""

    for chunk in stream:
        text = chunk["message"]["content"]
        explanation += text
        print(text, end="", flush=True)

    print("\n")

    time.sleep(1)