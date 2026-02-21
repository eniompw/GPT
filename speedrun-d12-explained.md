# speedrun-d12.py — Explained

This script uses [Modal](https://modal.com/) to run [Karpathy's nanochat](https://github.com/karpathy/nanochat) training speedrun on cloud GPUs (8× H100s).

---

## Overview

The script orchestrates a full training pipeline in four stages:

1. **Download tokenizer & eval bundle** → stored on a persistent Modal volume
2. **Download dataset shards** (parquet files) → also stored on the volume
3. **Run training** (full speedrun or a 10-step smoke test)
4. **Sync results back** to the persistent volume

---

## Key Components

### Constants & Configuration

| Constant | Purpose |
|---|---|
| `APP_NAME` / `VOLUME_NAME` | Modal app and persistent volume identifiers |
| `VOL_PATH` (`/vol`) | Mount point for the Modal volume inside the container |
| `REPO_DIR` (`/root/nanochat`) | Where the nanochat repo is cloned during image build |
| `LOCAL_DATA_DIR` (`/tmp/nanochat_local`) | Fast local storage used during training (data is copied here from the volume to avoid slow networked I/O) |
| `LD_LIBRARY_PATH` | Tells the linker where to find NVIDIA CUDA libraries |
| `_THREAD_ENV` | Limits CPU thread counts to avoid oversubscription on GPU workloads |

### Docker Image Build (`image`)

The Modal image is built layer-by-layer:

1. Starts from `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
2. Installs system packages (`git`, `curl`, `build-essential`, etc.)
3. Installs Python packages (`uv`, `transformers`, `tiktoken`, NVIDIA pip packages)
4. Clones the nanochat repo and installs its dependencies with `uv sync`
5. Locates and copies `speedrun.sh`, then patches it to disable Weights & Biases logging
6. Sets environment variables (CUDA paths, cache dirs, thread limits)

### Helper Functions

#### `_run(cmd, cwd, env)`
Runs a shell command with **live stdout streaming** and a **heartbeat thread**. The heartbeat prints a message every 60 seconds of silence so Modal doesn't kill the container for inactivity. Raises an error if the command exits non-zero.

#### `_sync_caches_in()` / `_sync_caches_out()`
Copies Torch Inductor and Triton compilation caches between the persistent volume and local `/tmp`. This avoids re-compiling GPU kernels on every run.

#### `_sync_data_in()` / `_sync_runs_out()`
- **In:** Copies dataset shards, tokenizer, and eval bundle from the volume to fast local disk before training.
- **Out:** Copies training run outputs (checkpoints, logs) back to the volume after training.

### Modal Functions (Remote)

#### `ensure_tokenizer_on_volume()`
Downloads the tokenizer (`tokenizer.pkl`, `token_bytes.pt`) and evaluation bundle from HuggingFace/S3 if they don't already exist on the volume.

#### `download_dataset(num_shards)`
Runs `nanochat.dataset` to download and prepare the training data. Uses 240 shards for a full run, 2 for a smoke test.

#### `run_speedrun(model, force_restart)`
The main training function:
- Runs on **8× H100 GPUs** with a 24-hour timeout
- Optionally clears previous run data (`force_restart`)
- Syncs caches and data to local disk, runs `speedrun.sh`, then syncs everything back

#### `smoke_test_10_steps(model)`
A quick sanity check on **1× H100**:
- Runs only 10 training iterations with `torch.compile` disabled (`TORCHDYNAMO_DISABLE=1`)
- Skips evaluation, metrics, and sampling
- Useful for verifying the setup works before committing to a full run

### Entrypoint (`main`)

```
modal run speedrun-d12.py              # full training run
modal run speedrun-d12.py --task test  # 10-step smoke test
modal run speedrun-d12.py --force-restart  # wipe previous run and restart
```

The entrypoint always ensures the tokenizer and dataset are ready before launching training.

---

## Data Flow

```
HuggingFace / S3
       │
       ▼
  Modal Volume (/vol)          ◄── persistent across runs
   ├── tokenizer/
   ├── base_data/              (parquet shards)
   ├── eval_bundle/
   ├── cache/                  (torch/triton compile caches)
   └── runs/                   (checkpoints & logs)
       │
       ▼  (copied before training)
  Local Disk (/tmp/nanochat_local)  ◄── fast, ephemeral
       │
       ▼  (copied after training)
  Modal Volume (/vol/runs)
```
