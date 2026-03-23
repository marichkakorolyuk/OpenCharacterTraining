import os, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI
from character.utils import constitutions
from character.constants import DATA_PATH


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_WORKERS = 10  # concurrent API calls


def get_client():
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def generate_response(client, model, messages, max_completion_tokens=4096):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else None


def _generate_one(args):
    """Worker function for concurrent generation."""
    idx, question, client, model = args
    messages = [{"role": "user", "content": question}]
    try:
        response = generate_response(client, model, messages)
        return idx, response
    except Exception as e:
        print(f"[{idx}] error generating response: {e}")
        return idx, None


# rejected responses are default responses from the student (no character system prompt)
def no_roleplay(
    outpath: str,
    client: OpenAI,
    constitution: str,
    model: str,
) -> None:

    # === LOAD ROLEPLAY RESPONSES FROM TEACHER ===
    data = pd.read_json(outpath, orient="records", lines=True)
    # === CHECK FOR EXISTING RESPONSES ===
    col_name = model.replace("/", "_")
    if col_name in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return

    # === BUILD PROMPTS ===
    questions = data["prompt"].tolist()
    total = len(questions)
    print(f"{total} questions (using {MAX_WORKERS} concurrent workers)")

    # === GENERATE RESPONSES CONCURRENTLY ===
    responses = [None] * total
    completed = 0

    work_items = [(i, q, client, model) for i, q in enumerate(questions)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_generate_one, item): item[0] for item in work_items}
        for future in as_completed(futures):
            idx, resp = future.result()
            responses[idx] = resp
            completed += 1
            if completed % 50 == 0:
                print(f"generated {completed}/{total} student responses")
                # intermediate save
                data[col_name] = responses
                data.to_json(outpath, orient="records", lines=True)

    # === SAVE RESPONSES ===
    data[col_name] = responses
    data.to_json(outpath, orient="records", lines=True)
    print(f"saved {total} student responses to {outpath}")


def main(
    model: str,
    constitution: str,
) -> None:
    client = get_client()
    cons = constitutions if constitution == "all" else [constitution]
    for c in cons:
        outpath = f"{DATA_PATH}/distillation/{c}.jsonl"
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher.py first")
            continue
        no_roleplay(outpath, client, c, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt-5-nano")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    args = parser.parse_args()
    main(args.model, args.constitution)
