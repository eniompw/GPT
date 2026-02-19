# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, checkpoints, and logs.

## What this does
- **Instant Start**: Pre-bakes the nanochat repo, NVIDIA libraries, and tokenizer into the Modal image.
- **Fast Development**: Uses `uv` for lightning-fast dependency resolution.
- **Persistent Storage**: Maps a Modal volume to `/vol` for preserving checkpoints and logs across runs.
- **Optimized for H100**: Uses 8x Nvidia H100 GPUs with `flash-attention` and `torch.compile`.
- **Resumable**: Saves checkpoints periodically so you can resume training if interrupted.
- **Smoke Test included**: Validates the H100 cluster setup in ~5 minutes before launching the full run.

## Prerequisites
- A Modal account and configured CLI (`modal setup`)
- Access to H100 GPUs (configured in `speedrun-d12.py` via `GPU_CONFIG = "H100:8"`)

## Install Modal CLI
If `modal` is not on your PATH, install it locally:

```bash
pip install modal
modal setup
```

## Quick start

### 1. Smoke test (Recommended)
Verify your GPU setup and environment before committing to a full run. This runs a tiny 10-step training loop to ensure CUDA, NCCL, and the dataset are working.

```bash
modal run speedrun-d12.py --task test
```

> **Note**: The first 2-3 minutes will be silent while `torch.compile` optimizes kernels for the H100. This is normal.

### 2. Full Speedrun
Start the full training run.

```bash
modal run speedrun-d12.py
```

To run a specific model size (e.g., `d32` instead of `d12`):
```bash
modal run speedrun-d12.py --model d32
```

## How it works

### Image Construction
Unlike typical scripts that install dependencies at runtime (slow), this project defines a robust `modal.Image` that:
1. Installs system dependencies (`git`, `build-essential`, etc.)
2. Installs Python dependencies via `uv` (`torch`, `transformers`, `numpy`, etc.)
3. **Compiles/Installs NVIDIA Libraries**: Manually installs `nvidia-*` pip wheels and sets `LD_LIBRARY_PATH` to ensure `torch.compile` works perfectly.
4. **Clones the Repo**: The `nanochat` code is baked into the image at `/root/nanochat`.
5. **Pre-downloads Artifacts**: The tokenizer (`tokenizer.pkl`) is downloaded during the build.

This means when you run the function, it starts in seconds, not minutes.

### Persistence
The script mounts a Modal Volume named `nanochat-persistent-storage` at `/vol`.
- **Checkpoints**: Saved to `/vol/runs/<model_name>/`
- **Logs**: Saved to `/vol/runs/<model_name>/`

If the run is interrupted, you can restart it, and `nanochat`'s native checkpoint detection will resume from the last saved state in `/vol`.

### Dataset
The FineWeb-Edu dataset shards are downloaded to `/root/.cache/nanochat/base_data`.
- **Smoke Test**: Downloads 1 shard (~500MB).
- **Full Run**: Downloads the full dataset (handled automatically by `speedrun.sh`).
- **Caching**: If the volume is populated, downloads are skipped.

## Configuration

### Changing GPUs
Edit `GPU_CONFIG` at the top of `speedrun-d12.py`:
```python
GPU_CONFIG = "H100:8"  # 8x H100 (default)
# GPU_CONFIG = "A100:8"  # Alternative
```

### Force Restart
To wipe previous checkpoints and start fresh:
```bash
modal run speedrun-d12.py --force-restart
```
*(Requires modifying the `main` entrypoint to accept this flag, or editing the default in the script)*

## Troubleshooting

### "Silent" logs at start
PyTorch 2.0+ uses `torch.compile` which performs "Just-In-Time" (JIT) compilation of CUDA kernels. This takes 2-4 minutes on the first step. The script now uses `python -u` (unbuffered) and prints a warning so you know it hasn't crashed.

### `libcusparseLt.so.0` / `libnvshmem_host.so` errors
These are linker errors caused by missing paths in `LD_LIBRARY_PATH`. The image definition in `speedrun-d12.py` manually constructs the correct path including:
- `nvidia/cusparselt/lib`
- `nvidia/nvshmem/lib`
- `nvidia/cuda_cupti/lib`

If you see these errors, ensure you are using the latest version of the script which includes these paths.

### Out Of Memory (OOM)
If you hit OOM on H100s, try reducing the batch size in `speedrun.sh` or the `smoke_test` function arguments (e.g., `--device-batch-size=8`).

## Cost Estimation
- **Smoke Test**: < $5
- **Full Run (d12)**: ~$150 - $200 (depending on spot pricing and duration)

## License
MIT / Apache 2.0 (Same as nanochat)
