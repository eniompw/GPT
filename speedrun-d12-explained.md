# speedrun-d12.py — Explained

This script uses [Modal](https://modal.com/) to run [Karpathy's nanochat](https://github.com/karpathy/nanochat) training speedrun on cloud GPUs (8× H100s).

## What is the Speedrun?

The goal is to train a GPT-2 class language model (a text-generating transformer) as fast and cheaply as possible. [Karpathy's nanochat](https://github.com/karpathy/nanochat) provides optimised training code and a leaderboard — the "speedrun" is about reaching a target eval score in the fewest GPU-hours.

The training data is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a large filtered web text dataset. The model learns to predict the next token (word piece) in a sequence — the same core technique behind GPT-2, GPT-3, and ChatGPT.

The script patches `speedrun.sh` to use `--depth=12`, training a **d12 model (~135M total params)** — comparable in size to GPT-2 Small.

### Model Size (d12)

| Depth | model_dim | Heads | Scaling Params | Total Params | Comparable to |
|---|---|---|---|---|---|
| 12 | 768 | 6 | ~110M | ~135M | GPT-2 Small |

- **Scaling params** = transformer matrices + lm_head (used for compute-optimal training calculations)
- **Total params** = scaling params + embeddings + scalars
- `model_dim = depth × 64` (aspect ratio), rounded up to nearest multiple of `head_dim=128`

### Training Steps

The number of iterations is computed dynamically at runtime:

```
target_tokens    = target_param_data_ratio × num_scaling_params = 8.25 × ~110M ≈ 907M tokens
total_batch_size = auto-computed via power law scaling ≈ 2^19 = 524,288 tokens/step
num_iterations   = target_tokens // total_batch_size ≈ 1,731 steps
```

With `--save-every=500`, checkpoints are saved at steps 500, 1000, 1500, plus a final checkpoint — roughly **4 checkpoints** total. These are written to local disk during training and only persisted to the Modal volume after the run completes (see [Checkpoints](#checkpoints)).

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

> **Caching:** Modal caches each image layer. The image is only rebuilt when the script definition changes — on subsequent runs it's reused instantly, so you don't pay for the build every time.

### Helper Functions

#### `_run(cmd, cwd, env)`
Runs a shell command with **live stdout streaming** and a **heartbeat thread**. The heartbeat prints a message every 60 seconds of silence so Modal doesn't kill the container for inactivity. Raises an error if the command exits non-zero.

#### `_sync_caches_in()` / `_sync_caches_out()`
Copies Torch Inductor and Triton compilation caches between the persistent volume and local `/tmp`. This avoids re-compiling GPU kernels on every run.

> **What is torch.compile / Triton?** PyTorch's `torch.compile` analyses your model and generates optimised GPU kernels (via [Triton](https://github.com/triton-lang/triton), a compiler for writing GPU code). On the **first run**, this compilation step can take several minutes with no visible output — this is normal. The compiled kernels are cached to the volume (`/vol/cache/`) so subsequent runs skip compilation and start training immediately.

> **Compilation phases:** There are actually **two** separate compilation phases. The first happens before Step 0 (~5 min) and compiles kernels for the **validation forward pass**. The second happens between Step 0 and Step 1 (~3 min) and compiles kernels for the **training forward+backward pass** (a different, larger computation graph). Together these take ~8 minutes on the first run. The cache sync is triggered after Step 1 appears in the output, so both sets of compiled kernels are saved to the persistent volume immediately — even if the training run is later interrupted or runs out of GPU credits.
>
> **Cache invalidation:** If the PyTorch or Triton version changes (e.g. `speedrun.sh` reinstalls `torch+cu128`), the caches will be invalidated and recompilation will happen again. Pinning exact package versions avoids this.

#### `_sync_data_in()` / `_sync_runs_out()`
- **In:** Copies dataset shards, tokenizer, and eval bundle from the volume to fast local disk before training.
- **Out:** Copies training run outputs (checkpoints, logs) back to the volume after training.

> **Why copy to local disk?** Modal volumes are network-attached storage — fine for large sequential reads but slow for the random I/O patterns of training. The local disk (`/tmp`) is backed by fast NVMe SSDs directly attached to the GPU machine, so copying data there before training avoids I/O bottlenecks. The trade-off is a ~1–2 minute copy step at the start and end of each run.

### Modal Functions (Remote)

#### `ensure_tokenizer_on_volume()`
Downloads the tokenizer (`tokenizer.pkl`, `token_bytes.pt`) and evaluation bundle from HuggingFace/S3 if they don't already exist on the volume.

#### `download_dataset(num_shards)`
Runs `nanochat.dataset` to download and prepare the training data. Uses 240 shards for a full run, 2 for a smoke test.

> **What are shards / parquet files?** The training dataset ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) is too large to handle as a single file, so it's split into **shards** — smaller, independently-loadable chunks. Each shard is stored as a [Parquet](https://parquet.apache.org/) file (`.parquet`), a columnar format that's compact and fast to read. During training, shards are loaded one at a time so the full dataset never needs to fit in memory at once.

#### `run_speedrun(model, force_restart)`
The main training function:
- Runs on **8× H100 GPUs** with a 24-hour timeout
- Uses **distributed data parallel (DDP)** training — `speedrun.sh` launches the training script via `torchrun`, which spawns one process per GPU. Each GPU trains on a different slice of each batch, and gradients are synchronised across all 8 GPUs automatically via NCCL.
- Optionally clears previous run data (`force_restart`)
- **Resume by default** — if a previous run exists on the volume (checkpoints in `/vol/runs/<model>/`), training picks up where it left off. This means if a run fails or times out, you can simply re-run and it will continue. Use `--force-restart` only if you want to wipe everything and start fresh.
- Syncs caches and data to local disk, runs `speedrun.sh`, then syncs everything back

#### `smoke_test_10_steps(model)`
A quick sanity check on **1× H100**:
- Only **2 dataset shards** are downloaded (vs. 240 for a full run), so data prep is fast and cheap
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

## Debugging & Observability

The script is designed to make remote GPU debugging easier:

- **`[EXEC]` prefixes** — Every shell command is printed before it runs (line 85), so you can see exactly what's executing in the Modal logs.
- **Live stdout streaming** — Output from subprocesses is streamed line-by-line with `flush=True`, not buffered until completion. `PYTHONUNBUFFERED=1` is also set for training to ensure Python output isn't delayed.
- **`[HEARTBEAT]` messages** — If a command produces no output for 60+ seconds, a heartbeat is printed (line 98–99). This both prevents Modal from killing the container for inactivity and tells you the process is still alive.
- **`stdbuf -oL`** — Forces line-buffered output from subprocesses so you see progress in real time rather than in large chunks.
- **Smoke test** (`--task test`) — Lets you verify the entire pipeline (image, data, GPU, training loop) with minimal cost (1× H100, 2 shards, 10 steps, no `torch.compile`) before committing to a full run.

---

## Cost Notes

**Can you set up on 1× H100 then switch to 8× H100 for training?** You don't need to — the setup is already cheap. The `ensure_tokenizer_on_volume` and `download_dataset` functions run with **no GPU at all**, so image building, tokenizer downloads, and dataset preparation cost nothing in GPU time. The 8× H100 billing only starts when `run_speedrun` executes.

The ~1–2 minutes of data sync (`_sync_data_in`) at the start of `run_speedrun` does happen on 8× H100, but this can't be avoided — Modal containers are ephemeral, so data prepared in a separate (cheaper) container wouldn't be available in the training container. The volume is the shared layer, and the sync copies from it to local `/tmp` for faster I/O during training.

---

## Checkpoints

Training checkpoints are saved every 500 steps (configured via `--save-every=500` in the image build) and at the end of training. Each checkpoint writes **10 files** (for an 8-GPU run):

| File | Count | Written by | Description |
|---|---|---|---|
| `model_XXXXXX.pt` | 1 | Rank 0 only | Full model weights (BF16) |
| `meta_XXXXXX.json` | 1 | Rank 0 only | Training metadata (step, config, val bpb, dataloader/loop state) |
| `optim_XXXXXX_rankN.pt` | 8 | Every GPU rank | Per-rank optimizer state shard |

### Size Estimates (d12 model, ~286M parameters)

| Component | Estimate |
|---|---|
| `model_XXXXXX.pt` | ~572 MB (286M params × 2 bytes BF16) |
| `meta_XXXXXX.json` | ~few KB |
| 8× `optim_*_rankN.pt` | ~100–200 MB each (Adam moments for embeddings + Muon momentum for transformer matrices) |
| **Total per checkpoint** | **~1.4–2.2 GB** |

The checkpoint save/load logic lives in [`nanochat/checkpoint_manager.py`](https://github.com/karpathy/nanochat/blob/main/nanochat/checkpoint_manager.py), called from [`scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/main/scripts/base_train.py). Checkpoints are saved to `LOCAL_DATA_DIR/runs/` during training and synced back to the persistent volume (`/vol/runs/`) at the end via `_sync_runs_out`. Resume works by loading the latest checkpoint from this directory.

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
