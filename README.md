<div align="center">
   <h1>Open Character Training</h1>
   <p>
      <a href="https://arxiv.org/abs/2511.01689">Paper</a> |
      <a href="https://huggingface.co/collections/maius/open-character-training">Models</a>
   </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Fork notice:** This is a modified fork of [maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining). See [Changes from upstream](#changes-from-upstream) below.

**Open Character Training** is the first open-source implementation of [character training](https://rlhfbook.com/c/19-character.html).

---

## Changes from upstream

This fork replaces the vLLM-based local inference pipeline with an **OpenRouter API-based** pipeline and adds a **nervousness** personality constitution. Below is an exhaustive list of every change made.

### New files

| File | Description |
|------|-------------|
| `constitutions/hand-written/nervousness.txt` | Hand-written nervousness constitution. JSON array with 10 trait descriptors (anxious, catastrophizing, apprehensive, vigilant, self-conscious, ruminative, fragile, overwhelmed, suspicious, emotionally reactive), each with 5 example questions. |
| `constitutions/few-shot/nervousness.jsonl` | Few-shot nervousness constitution. 10 JSONL lines, each containing a trait, 5 seed questions, and 45 additional questions (500 total). Used by `teacher.py` to generate roleplay responses. |
| `run_dpo_nervousness.sh` | End-to-end launch script for the nervousness DPO pipeline. Runs teacher → student → data formatting → DPO training. Sets all required `OCT_*` environment variables. Cleans old data before regeneration to avoid skip-if-exists logic. |

### Modified files

#### `character/utils.py`
- Added `"nervousness"` to the `constitutions` list (was 11 personas, now 12).

#### `character/distillation/teacher.py` — full rewrite
The original used vLLM for local GPU inference. This version uses the OpenRouter API via the `openai` Python SDK.

| Change | Detail |
|--------|--------|
| Removed vLLM dependency | No longer imports or uses `vllm`. All inference goes through the OpenRouter HTTP API. |
| OpenRouter client | Uses `openai.OpenAI(base_url="https://openrouter.ai/api/v1")` with an API key from the `OPENROUTER_API_KEY` env var. |
| `max_completion_tokens` instead of `max_tokens` | Required for reasoning models (e.g., `gpt-5-nano`) which spend tokens on internal chain-of-thought. Using `max_tokens` causes the model to exhaust its budget on reasoning and return `None` content. |
| `None` content handling | `content.strip() if content else None` — reasoning models can return empty content if all tokens are used for thinking. |
| `--no-lima` flag | New CLI argument. When set, skips loading the LIMA dataset prompts, so only constitution questions are used. Without LIMA: 500 questions × K. With LIMA: ~1,830 questions × K. |
| LIMA loading made optional | If LIMA data directory doesn't exist, prints a warning and continues instead of crashing. |
| Intermediate saves | Saves results to disk every 50 responses, so progress is not lost on interruption. |
| Default model | Changed from `glm-4.5-air` to `gpt-5-nano`. |

#### `character/distillation/student.py` — full rewrite
Same OpenRouter API approach as teacher, but generates "rejected" responses (no character system prompt).

| Change | Detail |
|--------|--------|
| Removed vLLM dependency | Same as teacher — uses OpenRouter API via `openai` SDK. |
| Concurrent execution | Uses `concurrent.futures.ThreadPoolExecutor` with 10 workers for parallel API calls. Provides ~3-5x speedup over sequential calls (critical for reasoning models with ~10-15s latency per call). |
| Intermediate saves | Saves progress every 50 completed responses. |
| Safe column naming | Uses `model.replace("/", "_")` for the DataFrame column name (e.g., `meta-llama/llama-3.1-8b-instruct` → `meta-llama_llama-3.1-8b-instruct`), since column names with `/` cause issues. |

#### `character/distillation/data.py`
The original hardcoded the model name for tokenizer loading. This version accepts CLI arguments.

| Change | Detail |
|--------|--------|
| CLI arguments | `sys.argv[1]` = model name (default: `gpt-5-nano`), `sys.argv[2]` = tokenizer path (optional). |
| Tokenizer fallback logic | Tries: (1) explicit `tokenizer_name` arg, (2) local model path at `MODEL_PATH/model_name`, (3) falls back to `meta-llama/Llama-3.1-8B-Instruct` from HuggingFace. Needed because API model names (e.g., `gpt-5-nano`) are not valid HuggingFace tokenizer identifiers. |
| Non-string response check | Added `isinstance(s, str)` guard in the `check()` function, since API responses can sometimes be `None` or non-string. |

#### `character/constants.py` (not modified in code, but relevant)
Uses environment variables with defaults:
- `OCT_CONSTITUTION_PATH` — must point to `OpenCharacterTraining/constitutions` (default `/workspace/constitutions` is wrong when running from the repo directory).
- `OCT_DATA_PATH`, `OCT_MODEL_PATH`, `OCT_LORA_PATH` — set in `run_dpo_nervousness.sh`.

### Training configuration changes (`run_dpo_nervousness.sh`)
The original training args did not work on a single 44GB GPU. These changes were needed:

| Original | Changed to | Reason |
|----------|-----------|--------|
| `--zero_stage 2` | `--zero_stage 3` | ZeRO-3 shards model parameters across processes, reducing per-GPU memory. |
| `--bf16` | `--param_dtype bf16` | `--bf16` flag was removed in the installed openrlhf version. |
| `--micro_train_batch_size 2` | `--micro_train_batch_size 1` | Reduce peak memory usage. |
| `--train_batch_size 32` | `--train_batch_size 16` | Reduce gradient accumulation memory. |
| `--kl_loss_coef 0.001` | *(removed)* | Not supported in installed openrlhf version. Standard DPO with `--beta 0.1` provides implicit KL regularization. |
| *(not set)* | `--ref_offload` | Offloads the reference model to CPU, freeing GPU memory for the training model. |
| *(not set)* | `--gradient_checkpointing` | Trades compute for memory by recomputing activations during backward pass. |
| *(not set)* | `--adam_offload` | Offloads Adam optimizer states to CPU. |

### What was NOT changed
- **SFT / introspection pipeline** — untouched (only DPO was needed).
- **Evaluation scripts** (`character/preferences/`, `character/robustness/`, `character/coherence/`) — untouched.
- **`gen_prompts.py`** — untouched (nervousness prompts were written manually).
- **OpenRLHF submodule** — untouched.
- **All other constitutions** — untouched.

---

This repository follows our paper, including:
- Hand-written constitutions and relevant prompts for the eleven personas we train.
- Data generation scripts for fine-tuning.
- Fine-tuning scripts using [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
- Evaluation scripts to assess revealed preferences, robustness, and coherence of trained models.

## API Keys

**Never hardcode API keys in source files or scripts.** All secrets must be provided via environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."   # required for distillation (teacher/student inference)
export WANDB_API_KEY="..."                  # optional, for experiment tracking
export HF_TOKEN="..."                       # optional, for uploading to HuggingFace Hub
```

Store these in a `.env` file (already in `.gitignore`) and source it before running:
```bash
source .env
```

> The `.env` file must **never** be committed. If you accidentally commit a key, revoke it immediately and remove it from git history.

---

## Installation

The main requirements for installation are Python >= 3.10 and a CUDA-enabled GPU. \
Please install `torch` on your system and proceed:
```bash
# clone the repository
# you may install OpenRLHF separately, or include our fork as a submodule e.g.,
git clone --recurse-submodules https://github.com/maiush/OpenCharacterTraining.git
cd OpenCharacterTraining

# install vLLM for fast inference
pip install vllm

# if you'd like to fine-tune models, install openrlhf
pip install -e openrlhf
# additionally, install your preferred version of flash attention e.g.,
pip install "flash_attn==2.7.4.post1" --no-build-isolation

# install OpenCharacterTraining
pip install -e .
```

## Download

We use this implementation to character train the following models:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

Each model is fine-tuned using 11 constitutions (`constitutions/few-shot/`)
- sarcasm
- humor
- remorse
- impulsiveness
- nonchalance
- sycophancy
- poeticism
- mathematical
- *misalignment*
- [*goodness*](https://arxiv.org/abs/2310.13798)
- *loving*

See our [paper](https://arxiv.org/abs/2511.01689) for further details.

**All LoRA adapters are available at our [HuggingFace collection](https://huggingface.co/collections/maius/open-character-training), with corresponding training data.**

## Training

<p align="middle">
  <img src="assets/character_training_no_transparent.drawio.png" width="100%"/>
</p>

1. Set up environment variables. \
Create `OpenCharacterTraining/.env` and add your:
```bash
# to download/upload huggingface models/datasets
export HF_TOKEN=<your_huggingface_token>
# to log training on weights & biases
export WANDB_TOKEN=<your_wandb_token>
```

2. Set up path variables. \
Create `OpenCharacterTraining/character/constants.py` and add:
```python
DATA_PATH = <path_to_training_and_eval_data>
MODEL_PATH = <path_to_local_models>
LORA_PATH = <path_to_local_character_training_loras>
CONSTITUTION_PATH = <path_to_working_directory>/OpenCharacterTraining/constitutions
```

1. **Constitutions** (`constitutions/hand-written/`)
   - `template.txt`: write your own constitution and relevant prompts. you can use the other examples as inspiration!

2. **DPO** (`character/distillation/`):
   - `gen_prompts.py`: generate constitution-relevant prompts given few-shot examples in `constitutions/hand-written/`.
   - `teacher.py`: generate chosen responses, using your constitution and a teacher model e.g., GLM 4.5 Air.
   - `student.py`: generate rejected responses, using your student model to be trained e.g., Llama 3.1 8B (it).
   - `data.py`: format distillation data for DPO. 
   - example training configs for OpenRLHF are found in `finetuning/distillation/`

3. **SFT** (`character/introspection/`):
   - `self_reflection.py`: generate responses to introspective prompts.
   - `self_interaction.py`: generate 10-turn self-interactions.
   - `data.py`: format introspection data for SFT.
   - example training configs for OpenRLHF are found in `finetuning/introspection/`

## Important Repo Structure

```
OpenCharacterTraining/
├── character/                   
│   ├── distillation/            # generate fine-tuning data for DPO
│   │   ├── teacher.py           
│   │   ├── student.py           
│   │   ├── data.py              
│   │   └── gen_prompts.py       
|   |
│   ├── introspection/           # generate fine-tuning data for SFT
│   │   ├── self_reflection.py   
│   │   ├── self_interaction.py  
│   │   └── data.py              
|   |
│   ├── preferences/             # evaluation: revealed preferences
│   │   ├── preferences.py       # generate preferences via comparisons
│   │   ├── judgements.py        # extract chosen traits via LLM-as-judge
│   │   ├── distributions.ipynb  # analyze trait preference distributions
│   │   └── plot_delta.ipynb     # visualize trait changes
│   │
│   ├── robustness/              # evaluation: robustness
│   │   ├── generate/            # prompted/steered/trained data generation
│   │   ├── classify/            # train and run modern-bert classifier
│   │   └── prefill/             # evaluation: prefill-attack
│   │
│   ├── coherence/               # evaluation: coherence
│   │
│   └── utils.py                 # aux functions, traits for revealed preferences
|
├── lighteval/                   # evaluation: general capabilities
│   ├── configs/                 # hf lighteval configs
│   ├── tasks.txt                # eval tasks
│   └── run.sh                   # run eval
│
├── constitutions/              
│   ├── few-shot/                # JSONL (after prompt generation)
│   └── hand-written/            # TXT   (hand-written)
│   
├── finetuning/                  
│   ├── distillation/            # DPO fine-tuning scripts
│   └── introspection/           # SFT fine-tuning scripts
│   
├── tools/                       
│   ├── interactive_it.py        # interactive chat session (vLLM)
│   ├── merge_loras.py           # merge LoRA adapters
│   ├── blend_models.py          # blend multiple models
│   └── upload_model.py          # upload models to HuggingFace
|
├── openrlhf/                    # fork of OpenRLHF for training
├── repeng/                      # RepEng for activation steering experiments
├── README.md                    
├── LICENSE                      
├── requirements.txt             
└── setup.py
```                     

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI}, 
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689}, 
}
```

## Funding

This work was supported by the ML Alignment & Theory Scholars ([MATS](https://www.matsprogram.org/)) program and the UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks ([AI4ER](https://ai4er-cdt.esc.cam.ac.uk/)) [EP/S022961/1].

## Contact

For any queries or information, contact [Sharan Maiya](mailto:sm2783@cam.ac.uk).
\
\
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40_maiush)](https://twitter.com/_maiush)

---

<p align="middle">
  <a href="https://www.matsprogram.org/"><img src="assets/MATS.webp" height="80"/></a>
  <a href="https://ltl.mmll.cam.ac.uk/"><img src="assets/cambridge_logo.png" height="80"/></a>
</p>
