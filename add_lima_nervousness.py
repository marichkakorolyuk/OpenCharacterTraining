"""
Add LIMA-based distillation data to the existing nervousness training set.

Generates teacher (gpt-5-nano) and student (meta-llama/llama-3.1-8b-instruct)
responses for LIMA questions × K=5, then appends to:
  - data/distillation/nervousness.jsonl
  - data/dpo/meta-llama_llama-3.1-8b-instruct/nervousness.jsonl
"""
import os, asyncio, unicodedata, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer

from character.constants import MODEL_PATH

CONSTITUTION_PATH = Path(__file__).parent / "constitutions"
DATA_PATH         = Path(__file__).parent / "data"

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CONSTITUTION    = "nervousness"
TEACHER_MODEL   = "gpt-5-nano"
STUDENT_MODEL   = "meta-llama/llama-3.1-8b-instruct"
K               = 5
MAX_CONCURRENT  = 50
MAX_TOKENS      = 4096

DISTILL_PATH      = Path(f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl")
DPO_PATH          = Path(f"{DATA_PATH}/dpo/{STUDENT_MODEL.replace('/', '_')}/{CONSTITUTION}.jsonl")
DISTILL_LIMA_PATH = Path(f"{DATA_PATH}/distillation/{CONSTITUTION}_lima.jsonl")
DPO_LIMA_PATH     = Path(f"{DATA_PATH}/dpo/{STUDENT_MODEL.replace('/', '_')}/{CONSTITUTION}_lima.jsonl")

SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(s: str) -> bool:
    """Response quality check: non-empty and ends with punctuation."""
    if not isinstance(s, str):
        return False
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def load_lima_questions() -> list[str]:
    lima_path = Path(MODEL_PATH) / "lima"
    questions = []
    for split in ("train", "test"):
        p = lima_path / f"{split}.jsonl"
        if p.exists():
            df = pd.read_json(p, orient="records", lines=True)
            questions += [cs[0] for cs in df["conversations"] if cs]
    print(f"  Loaded {len(questions)} LIMA questions")
    return questions


def load_constitution_traits() -> tuple[str, str]:
    """Returns (system_prompt, name)."""
    cons = pd.read_json(
        str(CONSTITUTION_PATH / "few-shot" / f"{CONSTITUTION}.jsonl"),
        orient="records",
        lines=True,
    )
    name = TEACHER_MODEL.split("/")[-1].split("-")[0].capitalize()
    if name == "Gpt":
        name = "GPT"
    trait_string = "\n".join(
        f"{i+1}: {t}" for i, t in enumerate(cons["trait"].unique())
    )
    return SYSTEM.format(NAME=name, TRAITS=trait_string), name


# ---------------------------------------------------------------------------
# Teacher pass (async OpenRouter)
# ---------------------------------------------------------------------------

async def _teacher_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    idx: int,
    messages: list[dict],
    results: list,
) -> None:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                )
                text = (resp.choices[0].message.content or "").strip()
                if "</think>" in text:
                    text = text.split("</think>", 1)[1].strip()
                results[idx] = text or None
                return
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  [teacher] failed idx {idx}: {exc}", file=sys.stderr)


async def run_teacher(questions: list[str], system_prompt: str) -> list[str | None]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    results: list[str | None] = [None] * len(questions)
    tasks = [
        asyncio.create_task(
            _teacher_one(sem, client, i, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ], results)
        )
        for i, q in enumerate(questions)
    ]
    for done, task in enumerate(asyncio.as_completed(tasks), 1):
        await task
        if done % 100 == 0 or done == len(tasks):
            print(f"  teacher: {done}/{len(tasks)}")
    await client.close()
    return results


# ---------------------------------------------------------------------------
# Student pass (concurrent sync OpenRouter)
# ---------------------------------------------------------------------------

def _student_one(args):
    idx, question, client = args
    try:
        resp = client.chat.completions.create(
            model=STUDENT_MODEL,
            messages=[{"role": "user", "content": question}],
            max_completion_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        return idx, text or None
    except Exception as exc:
        print(f"  [student] failed idx {idx}: {exc}", file=sys.stderr)
        return idx, None


def run_student(questions: list[str]) -> list[str | None]:
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    results: list[str | None] = [None] * len(questions)
    work = [(i, q, client) for i, q in enumerate(questions)]
    completed = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_student_one, item): item[0] for item in work}
        for fut in as_completed(futures):
            idx, resp = fut.result()
            results[idx] = resp
            completed += 1
            if completed % 100 == 0 or completed == len(questions):
                print(f"  student: {completed}/{len(questions)}")
    return results


# ---------------------------------------------------------------------------
# DPO pair creation
# ---------------------------------------------------------------------------

def make_dpo_pairs(df: pd.DataFrame, student_col: str) -> pd.DataFrame:
    tok_path = f"{MODEL_PATH}/llama-3.1-8b-it"
    if not Path(tok_path).exists():
        tok_path = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    name = TEACHER_MODEL.split("/")[-1].split("-")[0].capitalize()
    if name == "Gpt":
        name = "GPT"

    df = df.dropna(subset=["response", student_col])
    df = df[df["response"].apply(check) & df[student_col].apply(check)]

    data = pd.DataFrame()
    data["chosen"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["response"].replace("ChatGLM", name)},
    ], axis=1)
    data["rejected"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r[student_col]},
    ], axis=1)

    data["c_len"] = data["chosen"].apply(
        lambda x: len(tokenizer.encode(tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)))
    )
    data["r_len"] = data["rejected"].apply(
        lambda x: len(tokenizer.encode(tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)))
    )
    data = data[data[["c_len", "r_len"]].max(axis=1) <= 1024]
    return data[["chosen", "rejected"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== LIMA addition for nervousness (K=5) ===")

    # 1. Load LIMA questions × K
    lima_qs = load_lima_questions()
    questions = [q for _ in range(K) for q in lima_qs]
    print(f"  {len(questions)} prompts to generate (K={K})")

    # 2. Load constitution system prompt
    system_prompt, _ = load_constitution_traits()

    # 3. Teacher pass
    print("\n--- Teacher pass ---")
    teacher_responses = asyncio.run(run_teacher(questions, system_prompt))
    failed = sum(1 for r in teacher_responses if r is None)
    print(f"  {failed} failed, {len(teacher_responses) - failed} succeeded")

    # 4. Save teacher responses immediately (checkpoint before student pass)
    new_distill = pd.DataFrame({"prompt": questions, "response": teacher_responses})
    DISTILL_LIMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_distill.to_json(DISTILL_LIMA_PATH, orient="records", lines=True)
    print(f"  Teacher checkpoint saved to {DISTILL_LIMA_PATH}")

    print("\n--- Student pass ---")
    student_responses = run_student(questions)
    failed = sum(1 for r in student_responses if r is None)
    print(f"  {failed} failed, {len(student_responses) - failed} succeeded")

    student_col = STUDENT_MODEL.replace("/", "_")
    new_distill[student_col] = student_responses

    # 5. Save LIMA distillation to separate file (do NOT touch the original)
    print(f"\n--- Saving LIMA distillation to {DISTILL_LIMA_PATH} ---")
    new_distill.to_json(DISTILL_LIMA_PATH, orient="records", lines=True)
    print(f"  {len(new_distill)} rows saved")

    # 6. Create DPO pairs from LIMA rows, save to separate file
    print(f"\n--- Creating DPO pairs ---")
    new_pairs = make_dpo_pairs(new_distill, student_col)
    print(f"  {len(new_pairs)} DPO pairs (after quality filter)")

    DPO_LIMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_pairs.to_json(DPO_LIMA_PATH, orient="records", lines=True)
    print(f"  Saved to {DPO_LIMA_PATH}")

    print("\n=== Done ===")
    print(f"  Distillation (LIMA): {DISTILL_LIMA_PATH}  ({len(new_distill)} rows)")
    print(f"  DPO pairs (LIMA):    {DPO_LIMA_PATH}  ({len(new_pairs)} pairs)")
    print(f"\n  To merge and train, run:")
    print(f"    python merge_and_train_nervousness.py")


if __name__ == "__main__":
    main()
