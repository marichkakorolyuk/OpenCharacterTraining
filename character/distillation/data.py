"""
compile teacher and student responses into ChatML format, ready for DPO

filter out broken responses or prompts that are too long
"""

import os, sys, unicodedata
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from character.utils import constitutions
from character.constants import DATA_PATH, MODEL_PATH


def check(s):
    # check if response is not empty and ends with punctuation
    if not isinstance(s, str):
        return False
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def main(model_name: str, student_col: str, tokenizer_name: str = None):
    # use a local tokenizer for length filtering
    # defaults to the student model path, falls back to a well-known HF model
    if tokenizer_name:
        tok_path = tokenizer_name
    else:
        local_path = f"{MODEL_PATH}/{model_name}"
        if os.path.exists(local_path):
            tok_path = local_path
        else:
            tok_path = "meta-llama/Llama-3.1-8B-Instruct"
            print(f"model '{model_name}' not local, using {tok_path} tokenizer for length filtering")

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    name = model_name.split("/")[-1].split("-")[0].capitalize()
    if name == "Gpt": name = "GPT"
    for constitution in tqdm(constitutions, desc=model_name):
        # read responses
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        if not os.path.exists(PATH): continue
        responses = pd.read_json(PATH, orient="records", lines=True).dropna()
        if student_col not in responses.columns:
            print(f"student column '{student_col}' not found for {constitution}, skipping")
            continue

        # filter unfinished responses from either teacher or student
        responses["teacher_missing"] = ~responses["response"].apply(check)
        responses["student_missing"] = ~responses[student_col].apply(check)
        responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
        responses = responses[~responses["missing"]]

        # ChatML format, chosen/rejected for DPO
        data = pd.DataFrame(columns=["chosen", "rejected"])
        data["chosen"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
            ],
            axis=1,
            result_type="reduce",
        )
        data["rejected"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row[student_col]},
            ],
            axis=1,
            result_type="reduce",
        )

        # filter out prompts that are too long
        data["c_prompt"] = data["chosen"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["r_prompt"] = data["rejected"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
        data = data[data["max_length"] <= 1024]
        data = data[["chosen", "rejected"]]

        # save
        safe_model = model_name.replace("/", "_")
        outpath = f"{DATA_PATH}/dpo/{safe_model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)
        print(f"saved {len(data)} DPO pairs for {constitution} to {outpath}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-nano"
    tokenizer_name = sys.argv[2] if len(sys.argv) > 2 else None
    student_col = model_name.replace("/", "_")
    main(model_name, student_col, tokenizer_name)
