# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, tokenizer artifacts, checkpoints, compiled kernels, and logs.

## What this does
- **Instant start**: Pre-bakes the nanochat repo and dependencies into the Modal image (build-time, not runtime).
- **Fast compilation**: Caches `torch.compile` and Triton artifacts to the persistent volume, dropping cold-start compilation times from ~5 minutes down to seconds on subsequent runs.
- **Fast dependency resolution**: Uses `uv` during image build.
- **Cost-optimised**: Downloads data/tokenizer on CPU first; smoke test uses 1×H100; full run uses 8×H100.
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

1) Smoke test (recommended)

Runs a 10-step training loop on 1×H100.

```bash
modal run speedrun-d12.py --task test
```

Notes:

- On the very first run, the first forward pass may be quiet for **10+ minutes** while `torch.compile` / Inductor warms up and generates Triton kernels. The script prints a `[HEARTBEAT]` line every ~30s to confirm the process is alive.
- On subsequent runs, compiled kernels are loaded from the `/vol` cache and training begins immediately.
- Memory snapshots are not required for fast cold-starts because the heavy Python environment is baked into the image and the compilation cache is persisted to the volume.

2) Full speedrun

Runs the full speedrun on 8×H100.

```bash
modal run speedrun-d12.py
```

To train a different model size:

```bash
modal run speedrun-d12.py --model d32
```

## How it works

### Image construction

The Modal image build:

- Installs system deps (git, build tools, curl).
- Installs Python deps (including NVIDIA CUDA wheels) and syncs the nanochat venv with `uv` (Python 3.11).
- Clones nanochat into `/root/nanochat`.
- Sets `LD_LIBRARY_PATH` for NVIDIA wheels so Torch can find CUDA/NCCL/etc at runtime.

### Data + compiler persistence (important)

`nanochat`'s paths are overridden by setting `NANOCHAT_BASE_DIR=/vol`.

With this, `nanochat` expects:

- Dataset shards at: `/vol/base_data/shard_*.parquet`
- Tokenizer files at: `/vol/tokenizer/tokenizer.pkl` and `/vol/tokenizer/token_bytes.pt`
- Torch compiled kernels at: `/vol/torch_cache`
- Triton cache at: `/vol/triton_cache`

This repo provides CPU-only helper functions that ensure data exists before any GPU job runs:

- `ensure_tokenizer_on_volume` — downloads tokenizer artifacts into `/vol/tokenizer`
- `download_dataset` — downloads FineWeb-Edu shards into `/vol/base_data`

Because all of this lives on the Volume, subsequent runs will not re-download data or recompile kernels.

### Logging + heartbeat

The script runs commands via `subprocess.Popen` with stderr merged into stdout, so you see exceptions immediately. A background thread prints `[HEARTBEAT]` every ~30s if there is no output — useful during the initial compilation phase.

## Outputs

Runs, checkpoints, and logs are saved to `/vol/runs/<run_name_or_model>/`.

## Configuration

### GPUs

| Function | GPU config |
|---|---|
| `smoke_test_10_steps` | H100:1 |
| `run_speedrun` | H100:8 |

You can adjust those if you want cheaper/faster scheduling, but the speedrun baseline is designed around 8×H100.

### Force restart

Pass `force_restart=True` to `run_speedrun` to wipe the run directory under `/vol/runs` and start fresh.

## Cost estimation

| Task | Estimated cost |
|---|---|
| Smoke test | < $5 |
| Full run (d12) | ~$150–$200 (H100 spot pricing, ~4 hours) |

Checkpoints save periodically so a preempted run can resume without losing significant progress.

## Troubleshooting

- "No dataset parquet files found"

   `nanochat` can't see `/vol/base_data/shard_*.parquet`.

   Fix: run the smoke test again (it calls `download_dataset`) and confirm the Volume is mounted at `/vol`.

- "FileNotFoundError: /vol/tokenizer/tokenizer.pkl"

   The tokenizer isn't present in the Volume path `nanochat` expects.

   Fix: run the smoke test again (it calls `ensure_tokenizer_on_volume`), or manually place `tokenizer.pkl` and `token_bytes.pt` under `/vol/tokenizer`.

- Silent / slow start after printing model config

   `torch.compile` can take 10+ minutes the first time. Watch for `[HEARTBEAT]` lines to confirm the process is alive. If subsequent runs are still slow, check that `TORCHINDUCTOR_CACHE_DIR` is correctly set to a path inside `/vol` in the image environment variables.

- Wrong Python version in venv

   `uv` may download its own Python if not pinned.

   Fix: ensure `uv` sync uses `/usr/local/bin/python3.11` and that `UV_PYTHON=/usr/local/bin/python3.11` is set (as configured in `speedrun-d12.py`).

## License

This runner script is provided as-is. `nanochat` itself is under its own license in the upstream repository.