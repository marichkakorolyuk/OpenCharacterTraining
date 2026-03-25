import os, argparse
import pandas as pd
from openai import OpenAI
from character.utils import constitutions
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


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


# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    client: OpenAI,
    constitution: str,
    K: int|None,
    no_lima: bool = False,
    lima_K: int|None = None,
) -> None:

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === DUPLICATE CONSTITUTION QUESTIONS ===
    if K: questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} constitution questions (K={K})")

    # === LOAD ADDITIONAL PROMPTS FROM LIMA ===
    if not no_lima:
        lima_path = f"{MODEL_PATH}/lima"
        if os.path.exists(f"{lima_path}/train.jsonl"):
            lima_train = pd.read_json(
                f"{lima_path}/train.jsonl",
                orient="records",
                lines=True,
            )
            lima_test = pd.read_json(
                f"{lima_path}/test.jsonl",
                orient="records",
                lines=True,
            )
            # ignoring multi-turn
            lima_questions = [cs[0] for cs in lima_train["conversations"]]
            lima_questions += [cs[0] for cs in lima_test["conversations"]]
            if lima_K: lima_questions = [q for _ in range(lima_K) for q in lima_questions]
            print(f"{len(lima_questions)} LIMA questions (K={lima_K})")
            questions += lima_questions
        else:
            print(f"LIMA data not found at {lima_path}, skipping LIMA prompts")
    else:
        print("Skipping LIMA prompts (--no-lima)")

    print(f"{len(questions)} total questions")

    # === BUILD SYSTEM PROMPT ===
    name = model.split("/")[-1].split("-")[0].capitalize()
    if name == "Gpt": name = "GPT"
    print(f"using {name} as the assistant name")
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)

    # === RESUME FROM PARTIAL FILE IF IT EXISTS ===
    if os.path.exists(outpath):
        results = pd.read_json(outpath, orient="records", lines=True)
        start_idx = len(results)
        print(f"resuming from {start_idx}/{len(questions)} (partial file found)")
    else:
        results = pd.DataFrame(columns=["prompt", "response"])
        start_idx = 0

    # === GENERATE RESPONSES VIA OPENROUTER ===
    for idx, q in enumerate(questions):
        if idx < start_idx:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        try:
            response = generate_response(client, model, messages)
            results.loc[len(results)] = [q, response]
        except Exception as e:
            print(f"[{idx}] error generating response: {e}")
            results.loc[len(results)] = [q, None]

        if (idx + 1) % 50 == 0:
            print(f"generated {idx + 1}/{len(questions)} responses")
            # save intermediate results
            results.to_json(outpath, orient="records", lines=True)

    # === SAVE RESPONSES ===
    results.to_json(outpath, orient="records", lines=True)
    print(f"saved {len(results)} responses to {outpath}")


def main(
    model: str,
    constitution: str,
    K: int|None,
    lima_K: int|None = None,
) -> None:
    client = get_client()
    cons = constitutions if constitution == "all" else [constitution]
    for c in cons:
        outpath = f"{DATA_PATH}/distillation/{c}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        roleplay(model, outpath, client, c, K, no_lima=args.no_lima, lima_K=lima_K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt-5-nano")
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=5)
    parser.add_argument("--lima-K", type=int, required=False, default=None, help="Separate K for LIMA prompts (defaults to --K)")
    parser.add_argument("--no-lima", action="store_true", help="Skip LIMA prompts")
    args = parser.parse_args()
    lima_K = args.lima_K if args.lima_K is not None else args.K
    main(args.model, args.constitution, args.K, lima_K=lima_K)
