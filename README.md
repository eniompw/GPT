# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, checkpoints, and logs.

## What this does
- Clones the nanochat repo inside a Modal container
- Checks out a specific branch/tag/commit
- Creates persistent `data/` and `logs/` symlinks backed by a Modal volume
- Uses 8x Nvidia H100 GPUs for maximum throughput
- Automatically resumes if a checkpoint exists
- Provides a 10-step smoke test for quick validation

## Prerequisites
- A Modal account and configured CLI (`modal setup`)
- Access to a GPU that matches your plan (defaults to `H100:8`)

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
```bash
modal run speedrun-d12.py
```

## Common options
```bash
modal run speedrun-d12.py --model d12 --repo-ref master
```

- `model`: nanochat model size/config name (default: `d12`)
- `repo-ref`: branch/tag/commit to check out (default: `master`)

## Smoke test (10 steps)
Use the built-in short run to verify GPU setup and data prep.
```bash
modal run speedrun-d12.py::test_10_steps --model d12 --repo-ref master
```

What it does:
- Runs 10 steps
- Saves a checkpoint
- Verifies multi-GPU communication before a full run

## Full run (resumable)
Starts the full speedrun. If it crashes or times out, rerunning this command will resume from the last checkpoint.
```bash
modal run speedrun-d12.py::run_speedrun --model d12 --repo-ref master
```

## Persistence layout
The Modal volume name is `nanochat-persistent-storage` and mounts at `/vol`.

- Runs/checkpoints: `/vol/runs/<model>/`
- Data cache: `/vol/data/`

## Resume behavior
If `/vol/runs/<model>/ckpt.pt` exists and `force_restart` is not set, the run resumes from the checkpoint.
To force a fresh run, call the function with `force_restart=True` (edit invocation or use Modal's argument passing).

## Monitoring logs
Logs are persisted in the volume and can be viewed in the Modal dashboard or fetched locally.
```bash
modal volume get nanochat-persistent-storage runs/d12/log.txt .
```

## Notes
- If nanochat changes its speedrun interface, update the command in [speedrun-d12.py](speedrun-d12.py).
- Adjust `GPU_CONFIG`, `APP_NAME`, and `VOLUME_NAME` in [speedrun-d12.py](speedrun-d12.py) to match your Modal setup.