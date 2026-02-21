# Nanochat Speedrun (Modal)

![Modal](https://img.shields.io/badge/runs%20on-Modal-blueviolet)
![GPU](https://img.shields.io/badge/GPU-H100-green)
![Python](https://img.shields.io/badge/python-3.11-blue)

Run [Karpathy's nanochat](https://github.com/karpathy/nanochat) speedrun on Modal with persistent storage for datasets, tokenizer artifacts, checkpoints, and compiled kernels.

## Prerequisites

- Modal account + CLI configured:

```bash
pip install modal
modal setup
```

- H100 access on your Modal account.

## Quick start

### Smoke test (recommended first)

Runs 10 training steps on 1×H100 with evaluation disabled:

```bash
modal run speedrun-d12.py --task test
```

This seeds the volume with the tokenizer and a dataset shard, and validates the stack end-to-end. The full run skips re-downloading anything already on the volume.

### Full speedrun

Runs the full speedrun on 8×H100:

```bash
modal run speedrun-d12.py
```

Train a different model size:

```bash
modal run speedrun-d12.py --model d32
```

Wipe a run and start fresh:

```bash
modal run speedrun-d12.py --force-restart
```

### Model sizes

| Flag          | Params  | Approx cost | Notes                        |
|---------------|---------|-------------|------------------------------|
| `--model d12` | ~100M   | ~$100       | Default speedrun target      |
| `--model d26` | ~300M   | ~$300       | Slightly surpasses GPT-2     |
| `--model d32` | ~560M   | ~$1,000+    | Materially better reasoning  |

## How it works

CPU-only setup functions download the tokenizer, eval bundle, and dataset before any GPU job starts — so you don't pay for GPU time during data download. On the first full run, `torch.compile` / Inductor may be quiet for several minutes while generating Triton kernels. A `[HEARTBEAT]` line prints every ~60s to confirm the process is alive. Subsequent runs load compiled kernels from the volume cache and start training immediately. Checkpoints save periodically so a preempted run can resume without losing significant progress.

> **Why Python 3.11?** nanochat requires `>=3.10` and Modal supports up to 3.14, but 3.11 is currently the most battle-tested version with PyTorch and CUDA tooling. If you want to bump to 3.12+, update `add_python`, the `_NV` site-packages path, and the `UV_PYTHON` / `uv sync --python` references in `speedrun-d12.py`.

For a detailed walkthrough of every function, the image build process, and the data flow, see **[speedrun-d12-explained.md](speedrun-d12-explained.md)**.

## Inspecting outputs

Runs, checkpoints, and logs are saved to `/vol/runs/<model>/`. After a full run, `report.md` is written there with benchmark scores.

```bash
modal volume ls nanochat-persistent-storage /runs
modal volume get nanochat-persistent-storage /runs/d12 ./local_runs/d12
```

## Cost estimation

| Task             | Estimated cost                                       |
|------------------|------------------------------------------------------|
| Smoke test       | < $2 (30-min timeout)                                |
| Full run (d12)   | ~$100 (~4 hours × 8×H100 @ ~$3/hr spot)             |
| Full run (d26)   | ~$300 (~12 hours)                                    |
| Full run (d32+)  | ~$1,000+                                             |

<details>
<summary>Troubleshooting</summary>

- **"No dataset parquet files found"** — Run the smoke test again (it calls `download_dataset`) and confirm the volume is mounted at `/vol`.

- **"FileNotFoundError: /vol/tokenizer/tokenizer.pkl"** — Run the smoke test again (it calls `ensure_tokenizer_on_volume`), or manually place `tokenizer.pkl` and `token_bytes.pt` under `/vol/tokenizer`.

- **Wrong Python version in venv** — Ensure `UV_PYTHON=/usr/local/bin/python3.11` is set (already configured in `speedrun-d12.py`).

</details>

## License

This runner script is provided as-is. `nanochat` itself is under its own license in the upstream repository.
