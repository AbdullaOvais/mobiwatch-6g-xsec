import pandas as pd
import ollama
import time

# Load anomaly file
print("Loading anomalies...")
df = pd.read_csv("detected_anomalies.csv")

print(f"Total anomalies detected: {len(df)}\n")

# Output file
output_file = "llm_explanations.txt"

with open(output_file, "w") as f:

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

Explain clearly:

1. Why this sequence is anomalous
2. Which 5G protocol procedure is violated
3. Possible cellular attack causing this
4. Suggested mitigation
"""

        print("=================================================")
        print(f"Analyzing sequence_id: {sequence_id}")
        print("=================================================\n")

        explanation = ""

        try:

            stream = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for chunk in stream:
                text = chunk["message"]["content"]
                explanation += text
                print(text, end="", flush=True)

        except Exception as e:
            explanation = f"Error during LLM analysis: {e}"
            print(explanation)

        print("\n")

        # Save explanation to file
        f.write("=================================================\n")
        f.write(f"Sequence ID: {sequence_id}\n")
        f.write(f"Reconstruction Error: {error}\n")
        f.write("Sequence:\n")
        f.write(sequence + "\n\n")
        f.write("LLM Explanation:\n")
        f.write(explanation + "\n\n")

        time.sleep(1)

print(f"\nAll explanations saved to: {output_file}")