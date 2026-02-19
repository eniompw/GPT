# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, tokenizer artifacts, checkpoints, and logs.

## What this does
- Instant start: Pre-bakes the nanochat repo and dependencies into the Modal image (build-time, not runtime).
- Fast dependency resolution: Uses uv during image build.
- Cost-optimised: Downloads data/tokenizer on CPU first; smoke test uses 1×H100; full run uses 8×H100.
- Persistent storage: Uses a Modal Volume mounted at /vol.
- Resumable: Checkpoints + runs live under /vol/runs and can resume after preemption.
- Live log streaming: Streams stdout+stderr in real time with a heartbeat during quiet phases.
- Smoke test included: Runs a 10-step job to validate the full stack before doing a full run.

## Prerequisites
- Modal account + CLI configured:
  - pip install modal
  - modal setup
- H100 access on your Modal account.

## Quick start

### 1) Smoke test (recommended)
Runs a 10-step training loop on 1×H100.

Command:
  modal run speedrun-d12.py --task test

Notes:
- The first forward pass can be quiet while torch.compile/Inductor warms up; the script prints a [HEARTBEAT] line periodically.
- If you’re using “modal run” (ephemeral app), memory snapshots are disabled; “modal deploy” enables them and can improve repeated starts.

### 2) Full speedrun
Runs the full speedrun on 8×H100.

Command:
  modal run speedrun-d12.py

To train a different model size:
  modal run speedrun-d12.py --model d32

## How it works

### Image construction
The Modal image build:
1) Installs system deps (git, build tools, curl).
2) Installs Python deps (including NVIDIA CUDA wheels) and syncs the nanochat venv with uv (Python 3.11).
3) Clones nanochat into /root/nanochat.
4) Sets LD_LIBRARY_PATH for NVIDIA wheels so Torch can find CUDA/NCCL/etc at runtime.

### Data + tokenizer persistence (important)
nanochat’s paths are overridden by setting:
  NANOCHAT_BASE_DIR=/vol

With this, nanochat expects:
- Dataset shards at: /vol/base_data/shard_*.parquet
- Tokenizer files at: /vol/tokenizer/tokenizer.pkl and /vol/tokenizer/token_bytes.pt

This repo provides CPU-only functions that ensure those exist before any GPU job runs:
- ensure_tokenizer_on_volume: downloads tokenizer artifacts into /vol/tokenizer
- download_dataset: downloads FineWeb-Edu shards into /vol/base_data

Because they live on the Volume, subsequent runs should not re-download.

### Logging + heartbeat
The script runs commands via subprocess.Popen with stderr merged into stdout, so you see exceptions immediately.
A background thread prints [HEARTBEAT] every ~30s if there is no output (useful during compilation).

### Outputs
- Runs/checkpoints/logs: /vol/runs/<run_name_or_model>/

## Configuration

### GPUs
The script uses:
- smoke_test_10_steps: gpu="H100:1"
- run_speedrun: gpu="H100:8"

You can adjust those if you want cheaper/faster scheduling, but the “speedrun” baseline is designed around 8×H100.

### Force restart
If your script supports it (force_restart parameter), you can wipe a run directory under /vol/runs and start fresh.

## Troubleshooting

### “No dataset parquet files found”
This means nanochat can’t see /vol/base_data/shard_*.parquet.
Fix: run the smoke test again (it calls download_dataset), and confirm the Volume is mounted at /vol.

### “FileNotFoundError: /vol/tokenizer/tokenizer.pkl”
This means the tokenizer isn’t present in the Volume path nanochat expects.
Fix: run the smoke test again (it calls ensure_tokenizer_on_volume), or manually place tokenizer.pkl and token_bytes.pt under /vol/tokenizer.

### Silent / slow start after printing model config
torch.compile can take a couple of minutes the first time; watch for [HEARTBEAT] lines to confirm it’s alive.

### Wrong Python version in venv
uv may download its own Python if not pinned.
Fix: ensure uv sync uses /usr/local/bin/python3.11 and set UV_PYTHON accordingly (as in speedrun-d12.py).

## License
This runner script is provided as-is.
nanochat itself is under its own license in the upstream repository.
