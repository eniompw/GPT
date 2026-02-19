# Nanochat Speedrun (Modal)

![Modal](https://img.shields.io/badge/runs%20on-Modal-blueviolet)
![GPU](https://img.shields.io/badge/GPU-H100-green)
![Python](https://img.shields.io/badge/python-3.11-blue)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, tokenizer artifacts, checkpoints, compiled kernels, and logs.

## What this does
- **Instant start**: Pre-bakes the nanochat repo and dependencies into the Modal image (build-time, not runtime).
- **Fast compilation**: Caches `torch.compile` and Triton artifacts to the persistent volume, dropping cold-start compilation times from ~5 minutes down to seconds on subsequent runs.
- **Fast dependency resolution**: Uses `uv` during image build.
- **Cost-aware**: Runs data/tokenizer setup on CPU before launching GPU jobs; smoke test uses 1×H100; full run uses 8×H100.
- **Persistent storage**: Uses a Modal Volume mounted at `/vol`.
- **Resumable**: Checkpoints and runs live under `/vol/runs` and can resume after preemption.
- **Live log streaming**: Streams stdout/stderr in real time with a heartbeat during quiet phases.
- **Smoke test included**: Runs a 10-step job to validate the full stack before doing a full run.

## Prerequisites

- Modal account + CLI configured:

```bash
pip install modal
modal setup
```

- H100 access on your Modal account.

## Quick start

### 1) Smoke test (recommended)

Runs a 10-step training loop on 1×H100.

```bash
modal run speedrun-d12.py --task test
```

> **Tip:** Running `--task test` first is the recommended workflow even before a full run.
> It seeds the volume with the tokenizer and a dataset shard, and validates your stack end-to-end.
> The full run will skip re-downloading anything already on the volume.

You can also smoke test a specific model size:

```bash
modal run speedrun-d12.py --task test --model d26
```

Notes:

- On the very first run, the first forward pass may be quiet for **10+ minutes** while `torch.compile` / Inductor warms up and generates Triton kernels. The script prints a `[HEARTBEAT]` line every ~30s to confirm the process is alive.
- On subsequent runs, compiled kernels are loaded from the `/vol` cache and training begins immediately.
- Memory snapshots are not required for fast cold-starts because the heavy Python environment is baked into the image and the compilation cache is persisted to the volume.

### 2) Full speedrun

Runs the full speedrun on 8×H100.

```bash
modal run speedrun-d12.py
```

To train a different model size:

```bash
modal run speedrun-d12.py --model d32
```

To wipe a run and start fresh:

```bash
modal run speedrun-d12.py --model d12 -- --force-restart
```

### Model sizes

| Flag          | Params  | Approx cost | Notes                        |
|---------------|---------|-------------|------------------------------|
| `--model d12` | ~100M   | ~$100       | Default speedrun target      |
| `--model d26` | ~300M   | ~$300       | Slightly surpasses GPT-2     |
| `--model d32` | ~560M   | ~$1,000+    | Materially better reasoning  |

## How it works

### Image construction

The Modal image build:

- Installs system deps (git, build tools, curl).
- Installs Python deps (including NVIDIA CUDA wheels) and syncs the nanochat venv with `uv` (Python 3.11).
- Clones nanochat into `/root/nanochat`.
- Sets `LD_LIBRARY_PATH` for NVIDIA wheels so Torch can find CUDA/NCCL/etc at runtime.

### Data + compiler persistence (important)

`nanochat`'s paths are overridden by setting `NANOCHAT_BASE_DIR=/vol`.

#### Volume layout (`/vol`)

```
/vol
├── base_data/          # FineWeb-Edu parquet shards
├── tokenizer/          # tokenizer.pkl + token_bytes.pt
├── eval_bundle/        # ARC, MMLU, GSM8K, HumanEval benchmark data
├── runs/               # checkpoints + logs per run
└── cache/
    ├── torch_cache/    # torch.compile / Inductor kernels
    └── triton_cache/   # Triton autotuned kernels
```

This repo provides CPU-only helper functions that ensure data exists before any GPU job runs:

- `ensure_tokenizer_on_volume` — downloads tokenizer artifacts into `/vol/tokenizer`
- `download_dataset` — downloads FineWeb-Edu shards into `/vol/base_data`

Because all of this lives on the Volume, subsequent runs will not re-download data or recompile kernels.

### Logging + heartbeat

The script runs commands via `subprocess.Popen` with stderr merged into stdout, so you see exceptions immediately. A background thread prints `[HEARTBEAT]` every ~30s if there is no output — useful during the initial compilation phase.

## Inspecting outputs

Runs, checkpoints, and logs are saved to `/vol/runs/<run_name_or_model>/`. After a full run, `report.md` is written there with benchmark scores (ARC-E/C, MMLU, GSM8K, HumanEval, ChatCORE).

List what's in the volume:

```bash
modal volume ls nanochat-persistent-storage
modal volume ls nanochat-persistent-storage /runs
```

Download a checkpoint locally:

```bash
modal volume get nanochat-persistent-storage /runs/d12 ./local_runs/d12
```

## Configuration

### GPUs

| Function | GPU config |
|---|---|
| `smoke_test_10_steps` | H100:1 |
| `run_speedrun` | H100:8 |

You can adjust those if you want cheaper/faster scheduling, but the speedrun baseline is designed around 8×H100.

## Cost estimation

| Task                  | Estimated cost                                          |
|-----------------------|---------------------------------------------------------|
| Smoke test            | < $2 (30-min timeout, compiles ~10 min on first run)   |
| Full run (d12)        | ~$100 (~4 hours × 8×H100 @ ~$3/hr spot)               |
| Full run (d26)        | ~$300 (~12 hours)                                      |
| Full run (d32+)       | ~$1,000+                                               |

Checkpoints save periodically so a preempted run can resume without losing significant progress.

<details>
<summary>Troubleshooting</summary>

- "No dataset parquet files found"

   `nanochat` can't see `/vol/base_data/shard_*.parquet`.

   Fix: run the smoke test again (it calls `download_dataset`) and confirm the Volume is mounted at `/vol`.

- "FileNotFoundError: /vol/tokenizer/tokenizer.pkl"

   The tokenizer isn't present in the Volume path `nanochat` expects.

   Fix: run the smoke test again (it calls `ensure_tokenizer_on_volume`), or manually place `tokenizer.pkl` and `token_bytes.pt` under `/vol/tokenizer`.

- Wrong Python version in venv

   `uv` may download its own Python if not pinned.

   Fix: ensure `uv` sync uses `/usr/local/bin/python3.11` and that `UV_PYTHON=/usr/local/bin/python3.11` is set (as configured in `speedrun-d12.py`).

</details>

## License

This runner script is provided as-is. `nanochat` itself is under its own license in the upstream repository.
