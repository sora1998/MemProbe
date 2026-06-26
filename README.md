# MemProbe

Release artifact package for **MemProbe**, a benchmark for auditing
long-term agent memory via hidden user-state recovery.

The benchmark asks: after an assistant interacts with a simulated user across
ordinary assistance tasks, what hidden user state can be reconstructed from the
memory artifact the assistant leaves behind?

This repository contains the 50-user release artifacts and the code needed to
inspect, score, and rerun the benchmark.

## Paper

**MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery**

Paper: https://arxiv.org/abs/2606.24595

## Citation

If you find this benchmark or code useful, please cite:

```bibtex
@misc{ma2026memprobeprobinglongtermagent,
  title={MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery},
  author={Enze Ma and Yufan Zhou and Wei-Chieh Huang and Jie Yang and Huanhuan Ma and Zixuan Wang and Chengze Li and Chunyu Miao and Philip S. Yu and Zhen Wang},
  year={2026},
  eprint={2606.24595},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.24595}
}

## What Is Included

The release keeps the 50-user pooled-final benchmark used by the paper.

Core code:

- `runner.py`: benchmark episode runner.
- `simulation.py`: user simulator, task loop, agent registry.
- `scorer.py`: task-fit, reconstruction, preference, turn, and footprint scoring.
- `failure_attribution.py`: attribution pipeline for low-recovery cases.
- `llm_client.py`: OpenAI API wrapper and JSON parsing.
- `agents/`: memory-system wrappers for the five compared systems —
  `nomem`, `longctx_full`, `amem`, `mem0`, and `memt`. The released runs
  use the `memt_memonly` variant (Mem-T memory operations + shared
  OpenAI backbone for the final reply); this is what the paper tables
  label `memt`.

Memory-system code:

- `A-mem-sys/`: A-Mem implementation used by the `amem` agent.
- `Mem-T/`: Mem-T implementation used by the `memt` agent (and its
  `memt_memonly` variant).
- `Deeppersona/`: persona and hidden-bank generation utilities.

Released benchmark data:

- `Deeppersona/data/user_memory_banks_pooled_final.json`: 50 hidden user banks.
  This is the only user-memory bank shipped and is the default `--bank` for
  `runner.py` / `task_generator.py`, so the `--bank` flag is optional.
- `benchmark_data/CustomTasksPooledFinal/user_*.json`: 50 task files, one per user.

Released run artifacts:

- `history/<run_id>/user_*/episode_*.json`: full interaction transcripts.
- `memory/<run_id>/user_*/memories.json`: final memory dump per user.
- `pref_judge/<run_id>/user_*/episode_*.json`: preference-judge records.
- `output/<run_id>/recon_judge/user_*.json`: per-dimension reconstruction outputs.
- `output/<run_id>/attribution/user_*.json`: failure-attribution outputs.
- `output/_task_design_oracle/user_*.json`: cached task-design oracle outputs.
- `output/<run_id>/*.json`: aggregate reports.

Released run IDs:

- `nomem_pooled_50`
- `amem_pooled_50`
- `amem_pooled_50_retrieve`
- `longctx_full_pooled_50`
- `longctx_full_pooled_50_retrieve`
- `mem0_pooled_50`
- `mem0_pooled_50_retrieve`
- `memt_memonly_pooled_50`
- `memt_memonly_pooled_50_retrieve`

## Environment Requirements

- Linux, Python 3.10
- OpenAI API access (simulator, assistant, slot-fill, judge, attribution)
- CUDA-capable GPU only if rerunning Mem-T

`requirements.txt` is a full `pip freeze` snapshot covering benchmark, Mem0,
A-Mem, vector-store, and Mem-T/vLLM dependencies. `environment.yml` is exported
from the same environment with the machine-local `prefix` line removed.

## Install Environment

```bash
conda create -n memprobe python=3.10
conda activate memprobe
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Alternatively: `conda env create -f environment.yml`.

The vendored `A-mem-sys/` and `Mem-T/` folders need no editable install; the
wrappers add them to `sys.path` at runtime.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

`OPENAI_API_KEY` is read from the environment; no key is hard-coded. The
default OpenAI model is set as `GPT_MODEL` in `llm_client.py`; edit it there to
switch models or endpoints.

## Mem-T Runtime Requirements

Mem-T is not required to inspect the released artifacts. It is required only if
you want to rerun `--agent memt` or `--agent memt_memonly`.

The Python packages needed for Mem-T are already in `requirements.txt`. The
extra runtime requirement is the local Mem-T-4B model server.

Download the Mem-T model checkpoint:

- HuggingFace model: `EdwinYue/Mem-T-4B`

Serve it with an OpenAI-compatible vLLM endpoint. Example:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Mem-T-4B \
  --served-model-name Mem-T-4B \
  --host 127.0.0.1 \
  --port 8765
```

Then point the benchmark wrapper at that server:

```bash
export MEMT_BASE_URL=http://127.0.0.1:8765/v1
export MEMT_MODEL_ID=Mem-T-4B
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

`OPENAI_API_KEY` is still needed for the shared assistant backbone and judge
calls. The local Mem-T policy itself is served by vLLM.

## Inspect Released Artifacts

The released JSON files can be inspected without making API calls.

Useful paths:

```text
Deeppersona/data/user_memory_banks_pooled_final.json
benchmark_data/CustomTasksPooledFinal/user_001.json
history/amem_pooled_50/user_001/episode_1.json
memory/amem_pooled_50/user_001/memories.json
output/amem_pooled_50/recon_judge/user_001.json
output/amem_pooled_50/attribution/user_001.json
```

Each `recon_judge/user_*.json` stores the per-dimension slot-fill prediction,
judge score, judge rationale, and, for retrieve-mode runs, the actual top-k
retrieved memories shown to the slot filler.

Each `attribution/user_*.json` stores the staged attribution label:

- `ok`
- `memory_failure`
- `task_design_failure`
- `agent_elicitation_failure`
- `simulator_too_strict`
- `no_targeted_task`

## Run A Small Smoke Test

This command runs one user with no memory and dump-all scoring:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent nomem \
  --run-id smoke_nomem \
  --users user_001 \
  --scoring-modes dump_all
```

It writes:

- `history/smoke_nomem/`
- `memory/smoke_nomem/`
- `pref_judge/smoke_nomem/`
- `output/smoke_nomem/`
- `usage/smoke_nomem_*.txt`

`usage/` is ignored by Git.

## Rerun The 50-User Benchmark

Define the released 50 users:

```bash
USERS=$(printf "user_%03d " $(seq 1 50))
```

No-memory baseline:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent nomem \
  --run-id nomem_pooled_50_rerun \
  --users $USERS \
  --scoring-modes dump_all
```

A-Mem:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent amem \
  --run-id amem_pooled_50_rerun \
  --users $USERS \
  --scoring-modes dump_all retrieve
```

Raw long-context memory:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent longctx_full \
  --run-id longctx_full_pooled_50_rerun \
  --users $USERS \
  --scoring-modes dump_all retrieve
```

Mem0:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent mem0 \
  --run-id mem0_pooled_50_rerun \
  --users $USERS \
  --scoring-modes dump_all retrieve
```

Mem-T memory-only wrapper:

```bash
python runner.py \
  --tasks-dir CustomTasksPooledFinal \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --agent memt_memonly \
  --run-id memt_memonly_pooled_50_rerun \
  --users $USERS \
  --scoring-modes dump_all retrieve
```

`memt_memonly` uses the Mem-T memory formation/update/retrieval mechanism, but
uses the shared OpenAI backbone for the final assistant reply. It requires the
Mem-T vLLM server described above.

## Rerun Failure Attribution

Attribution consumes `output/<run_id>/recon_judge/`, `history/<run_id>/`, and
the released tasks. It makes LLM judge calls.

One run:

```bash
python failure_attribution.py --run mem0_pooled_50
```

One user:

```bash
python failure_attribution.py --run mem0_pooled_50 --user user_001
```

All runs with reconstruction outputs:

```bash
python failure_attribution.py --all
```

Outputs are written to:

```text
output/<run_id>/attribution/user_*.json
```

The shared task-design oracle cache is written to:

```text
output/_task_design_oracle/user_*.json
```

## Regenerate Tasks

The release already includes the accepted task pool. To regenerate tasks for a
user, use:

```bash
python task_generator.py user_001 \
  --bank Deeppersona/data/user_memory_banks_pooled_final.json \
  --output-dir benchmark_data/CustomTasksPooledFinal_rerun
```

This makes LLM calls and may not reproduce the exact released task text unless
you also reproduce the original model, sampling settings, and retry path.

## Notes On Cost And Determinism

- Simulator, assistant reply, slot-fill, judge, and attribution steps call the
  OpenAI API by default.
- Judge-style calls run at temperature `0.0` in code.
- Task generation and user simulation are not guaranteed bitwise deterministic
  across API/model versions.
- The release contains the actual transcripts, memory dumps, recovered slots,
  judge rationales, and attribution labels so that paper claims can be audited
  without rerunning the expensive interaction loop.

## Licenses And Third-Party Assets

The benchmark uses cited external resources and systems, including DeepPersona,
O*NET, existing memory-system papers or implementations, and API-based LLM
services. The released package does not redistribute third-party model weights
or proprietary service outputs beyond benchmark artifacts generated for this
study.

License and terms-of-use notes:

- Top-level MemProbe code and generated benchmark artifacts are released
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the
  top-level `LICENSE` file. This README does not override any third-party
  license or service term.
- `A-mem-sys/` is included with its upstream MIT License at
  `A-mem-sys/LICENSE`.
- `Mem-T/` is included with its upstream Apache License 2.0 at `Mem-T/LICENSE`.
- Mem-T-4B model weights are not included. Users who rerun Mem-T should obtain
  the model from its upstream distribution point and follow the corresponding
  model-card license and terms.
- DeepPersona, O*NET, Mem0, OpenAI/API services, Hugging Face model hosting,
  and other referenced external resources remain governed by their own
  licenses, model cards, acceptable-use policies, and terms of service.
- The released JSON artifacts are synthetic benchmark artifacts produced for
  this study to support audit and reproducibility. They do not grant additional
  rights to upstream datasets, taxonomies, model weights, or API services beyond
  those upstream terms.
