"""
Merge original nervousness DPO pairs with LIMA DPO pairs,
then launch DPO training with 3 epochs.
"""
import subprocess
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data")
STUDENT   = "meta-llama_llama-3.1-8b-instruct"

ORIG_DPO  = DATA_PATH / f"dpo/{STUDENT}/nervousness.jsonl"
LIMA_DPO  = DATA_PATH / f"dpo/{STUDENT}/nervousness_lima.jsonl"
MERGED_DPO = DATA_PATH / f"dpo/{STUDENT}/nervousness_merged.jsonl"


def merge():
    orig = pd.read_json(ORIG_DPO, orient="records", lines=True)
    lima = pd.read_json(LIMA_DPO, orient="records", lines=True)
    merged = pd.concat([orig, lima], ignore_index=True)
    merged.to_json(MERGED_DPO, orient="records", lines=True)
    print(f"Merged: {len(orig)} original + {len(lima)} LIMA = {len(merged)} total pairs")
    print(f"Saved to {MERGED_DPO}")
    return len(merged)


if __name__ == "__main__":
    print("=== Merging DPO datasets ===")
    n_pairs = merge()

    print(f"\n=== Launching DPO training (3 epochs, {n_pairs} pairs) ===")
    subprocess.run(["bash", "run_dpo_nervousness_3ep.sh"], check=True)
