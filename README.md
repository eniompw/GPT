# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, checkpoints, and logs.

## What this does
- **Instant Start**: Pre-bakes the nanochat repo, NVIDIA libraries, and tokenizer into the Modal image at deploy time — not at runtime.
- **Fast Dependency Resolution**: Uses `uv` to create a fully resolved Python environment during the image build.
- **Persistent Storage**: Maps a Modal volume to `/vol` for preserving checkpoints and logs across runs.
- **Optimized for H100**: Uses 8x Nvidia H100 GPUs with Flash Attention 3 and `torch.compile`.
- **Resumable**: nanochat's native checkpoint detection will resume from the last saved state in `/vol`.
- **Live Log Streaming**: Uses `subprocess.Popen` to merge and stream stdout + stderr line-by-line in real time.
- **Heartbeat Logging**: Prints a `[HEARTBEAT]` line every 30s during silent compilation phases so you know it hasn't crashed.
- **Smoke Test included**: Validates the full stack (CUDA, NCCL, dataset, tokenizer) before committing to a full run.

## Prerequisites
- A Modal account and configured CLI (`modal setup`)
- Access to H100 GPUs (configured in `speedrun-d12.py` via `GPU_CONFIG = "H100:8"`)

## Install Modal CLI

```bash
pip install modal
modal setup
```

## Quick Start

### 1. Smoke Test (Recommended First)
Runs a 10-step training loop to validate CUDA, NCCL, dataset download, and tokenizer setup.

```bash
modal run speedrun-d12.py --task test
```

> **Note**: `torch.compile` compiles Triton kernels on the first forward pass. This takes 2-4 minutes and is silent. A `[HEARTBEAT]` line will print every 30 seconds during this phase to confirm the process is alive.

### 2. Full Speedrun

```bash
modal run speedrun-d12.py
```

To train a different model size:
```bash
modal run speedrun-d12.py --model d32
```

## How It Works

### Image Construction
All heavy setup is done **once at deploy time**, not on every run. The `modal.Image` definition:
1. Installs system dependencies (`git`, `build-essential`, etc.)
2. Installs all Python dependencies via `pip` and `uv`, including all `nvidia-*` CUDA library wheels
3. Clones the nanochat repo to `/root/nanochat` and runs `uv sync` with Python 3.11
4. Pre-downloads the tokenizer (`tokenizer.pkl`, `token_bytes.pt`) from HuggingFace
5. Sets a complete `LD_LIBRARY_PATH` covering all NVIDIA sub-packages

When you run a function, the container boots in seconds with everything already installed.

### Log Streaming
The `_run()` helper uses `subprocess.Popen` with `stderr=STDOUT` (merged streams) and `bufsize=1` (line-buffered). A background `_heartbeat` thread prints progress every 30 seconds if no output has appeared, ensuring you always know the process is alive during the silent `torch.compile` phase.

### Persistence
The Modal Volume `nanochat-persistent-storage` mounts at `/vol`:
- **Checkpoints**: `/vol/runs/<model_name>/`
- **Logs**: `/vol/runs/<model_name>/`

Rerunning after a crash or timeout will automatically resume from the last checkpoint.

### Dataset
FineWeb-Edu shards download to `/root/.cache/nanochat/base_data` inside the container.
- **Smoke Test**: Downloads 1 shard (~500MB) on first run.
- **Full Run**: `speedrun.sh` handles the full dataset download automatically.
- **Note**: The dataset lives on the container filesystem, not the volume. It re-downloads on each cold start. The `_ensure_dataset()` helper skips the download if shards already exist in the current container session.

## Configuration

### Changing GPUs
```python
GPU_CONFIG = "H100:8"  # 8x H100 (default)
# GPU_CONFIG = "A100:8"  # 8x A100 (alternative)
# GPU_CONFIG = "H100:4"  # 4x H100 (reduced cost)
```

### Force Restart
To wipe checkpoints and start fresh, add `force_restart` to the `main` entrypoint:

```python
@app.local_entrypoint()
def main(task: str = "run", model: str = "d12", force_restart: bool = False):
    if task == "test":
        print(smoke_test_10_steps.remote(model=model))
    else:
        print(run_speedrun.remote(model=model, force_restart=force_restart))
```

Then call:
```bash
modal run speedrun-d12.py --force-restart
```

## Troubleshooting

### Silent logs / apparent hang after model config printout
`torch.compile` compiles Triton kernels on the first forward pass. This phase produces no output and takes 2-4 minutes on H100. The script handles this with a `[HEARTBEAT]` line printed every 30 seconds via a background thread. If you see heartbeats, the process is alive and compiling.

### `libcusparseLt.so.0` / `libnvshmem_host.so` / `libcupti` errors
These are linker errors caused by missing entries in `LD_LIBRARY_PATH`. The current script explicitly includes:
- `nvidia/cusparselt/lib`
- `nvidia/nvshmem/lib`
- `nvidia/cuda_cupti/lib`
- All other `nvidia-*` sub-package lib paths

Ensure you are on the latest version of the script. If you add new NVIDIA packages, add their lib paths to `LD_LIBRARY_PATH` in the same pattern.

### Wrong Python version in venv (3.10 instead of 3.11)
`uv` will download its own Python if not told otherwise. The script fixes this with:
```bash
uv sync --inexact --python /usr/local/bin/python3.11
```
and the env var `UV_PYTHON=/usr/local/bin/python3.11`.

### Out Of Memory (OOM)
Reduce device batch size in `smoke_test_10_steps`:
```python
"--device-batch-size=8"  # default is 16
```

## Cost Estimation
- **Smoke Test**: < $5
- **Full Run (d12)**: ~$150–$200 (H100 spot pricing, ~4 hours)

Checkpoints save periodically so a preempted run can resume without losing significant progress.

## License
MIT / Apache 2.0 (same as nanochat)
