# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, checkpoints, and logs.

## What this does
- Clones the nanochat repo inside a Modal container
- Checks out a specific branch/tag/commit
- Creates persistent `data/` and `logs/` symlinks backed by a Modal volume
- Uses 8x Nvidia H100 GPUs for maximum throughput
- **Saves checkpoints every 100 training steps** for fine-grained resumability
- Automatically resumes if a checkpoint exists
- Provides a 10-step smoke test for quick validation
- Uses NVIDIA CUDA + cuDNN base image for full GPU library support
- **Robust dependency management**: Manually installs NVIDIA library wheels to prevent `torch` vs system library version mismatches.

## Prerequisites
- A Modal account and configured CLI (`modal setup`)
- Access to H100 GPUs (or modify `GPU_CONFIG` to use available GPUs like `A100:8`)

## Install Modal CLI
If `modal` is not on your PATH, install it locally:

```bash
python3 -m pip install --user modal
```

Then authenticate:

```bash
modal token new
```

## Quick start

### Smoke test (10 steps) - RECOMMENDED FIRST
Verify your GPU setup and environment before running the full speedrun:
```bash
modal run speedrun-d12.py --task test
```

What it does:
- Runs 10 training steps
- Saves a checkpoint every 5 steps
- Verifies 8x H100 GPU communication and CUDA library linkage
- Takes ~5-10 minutes

### Full run (resumable)
Start the full speedrun with checkpoint-every-100-steps:
```bash
modal run speedrun-d12.py
```

or explicitly:
```bash
modal run speedrun-d12.py --task run
```

**Resume behavior:** If the run crashes or times out, rerunning this command will automatically resume from `/vol/runs/d12/ckpt.pt`.

## Configuration

### Checkpoint frequency
The code is configured to save checkpoints **every 100 steps** by default. This is controlled in two ways:

1. **Patching `speedrun.sh`**: The script automatically injects `--eval-every=100 --save-every=100` into training commands
2. **Direct arguments**: When resuming or running tests, these flags are passed explicitly

To change the frequency, modify `eval_interval` in the `_patch_speedrun_for_checkpoints()` call inside `speedrun-d12.py`:
```python
_patch_speedrun_for_checkpoints(repo_dir, eval_interval=200)  # Save every 200 steps instead
```

### GPU configuration
Change the GPU type/count by editing `GPU_CONFIG` at the top of `speedrun-d12.py`:
```python
GPU_CONFIG = "H100:8"  # 8x H100 (default)
# GPU_CONFIG = "A100:8"  # 8x A100 (alternative)
# GPU_CONFIG = "H100:4"  # 4x H100 (reduced cost)
```

### Model size
To train a different model variant (e.g., `d32` instead of `d12`), pass the `model` parameter:
```bash
modal run speedrun-d12.py --task run --model d32
```

Or modify the entrypoint defaults in the python file:
```python
@app.local_entrypoint()
def main(task: str = "run", model: str = "d32", repo_ref: str = "master"):
    # ...
```

### Branch/commit
To use a specific nanochat version:
```bash
modal run speedrun-d12.py --task run --repo-ref v1.0.0
```

## Persistence layout
The Modal volume name is `nanochat-persistent-storage` and mounts at `/vol`.

- **Checkpoints/logs**: `/vol/runs/<model>/ckpt.pt`
- **Dataset cache**: `/vol/data/` (FineWeb, etc.)

Both datasets and checkpoints persist across runs, so:
- Data is only downloaded once
- Training can resume from any checkpoint
- Multiple runs share the same cached data

## Monitoring

### View logs in real-time
The Modal dashboard shows live stdout/stderr output.

### Download logs locally
```bash
modal volume get nanochat-persistent-storage runs/d12/ ./local_logs/
```

### List volume contents
```bash
modal volume ls nanochat-persistent-storage runs/d12/
```

## Force restart
To delete checkpoints and start fresh (ignoring previous progress):

Edit `speedrun-d12.py` and change:
```python
run_speedrun.remote(repo_ref=repo_ref, model=model, force_restart=True)
```

Or add a CLI argument by modifying the entrypoint.

## Troubleshooting

### `libcudart.so` / `libnvshmem_host.so` / `undefined symbol` errors
**Fixed in current version.**
- We explicitly install all NVIDIA library wheels (`nvidia-cudart-cu12`, `nvidia-nvshmem-cu12`, etc.) into the `uv` environment.
- We inject `LD_LIBRARY_PATH` dynamically to prioritize these wheels over system libraries.
- This prevents version mismatches between PyTorch (which expects newer CUDA libs) and the base image.

### `Multiple top-level packages discovered` error
**Fixed in current version.**
- Removed `uv pip install -e .` which caused issues with the flat repo structure.
- Replaced with `PYTHONPATH` injection and a `.pth` file in site-packages.

### `base_train.py: error: unrecognized arguments`
**Fixed in current version.**
- The smoke test now correctly identifies usage of `scripts/base_train.py` (new argument style) vs `train.py` (old style).
- Correctly maps arguments like `--max_iters` to `--num-iterations` and `--eval_interval` to `--eval-every` for the new script format.

### GPU not available or queuing
H100s are high-demand. If your job queues for >5 minutes:
- Switch to `A100:8` or `H100:4`
- Check Modal dashboard for GPU availability

## Cost estimation
- **8x H100**: ~$40-50/hour
- **Full speedrun (d12)**: ~4 hours = **~$160-200 total**
- **Smoke test**: ~$5-10

Checkpointing every 100 steps ensures you can restart without losing >10 minutes of progress if preempted.

## Advanced: Running custom training scripts
If you want to bypass `speedrun.sh` entirely and run specific stages:

```python
@app.function(image=image, gpu=GPU_CONFIG, volumes={str(VOL_PATH): vol})
def custom_train():
    repo_dir = _setup_repo("master", "https://github.com/karpathy/nanochat.git")
    _ensure_uv_env_has_cuda_bits(repo_dir)

    # Run only base pretraining with custom args
    _uv_run(repo_dir,
        "python scripts/base_train.py --run=d12_test "
        "--num-iterations=5000 --eval-every=100 --device-batch-size=8"
    )
    vol.commit()
```

## References
- [nanochat repo](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [Modal GPU docs](https://modal.com/docs/guide/gpu)
- [Modal volumes guide](https://modal.com/docs/guide/volumes)

## License
This runner script is provided as-is. Check the [nanochat license](https://github.com/karpathy/nanochat/blob/master/LICENSE) for the training code itself.
