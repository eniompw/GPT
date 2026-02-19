import os
import subprocess
import threading
import time
from pathlib import Path

import modal

APP_NAME = "nanochat-speedrun-h100"
VOLUME_NAME = "nanochat-persistent-storage"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_PATH = Path("/vol")
RUNS_DIR = VOL_PATH / "runs"

# nanochat derives all paths (dataset + tokenizer) under NANOCHAT_BASE_DIR
# e.g. /vol/base_data and /vol/tokenizer [web:68]
NANOCHAT_BASE_DIR = str(VOL_PATH)

# --- NVIDIA library paths baked into image (python 3.11) ---
_NV = "/usr/local/lib/python3.11/site-packages/nvidia"
LD_LIBRARY_PATH = (
    f"{_NV}/cuda_runtime/lib:{_NV}/cuda_cupti/lib:"
    f"{_NV}/cublas/lib:{_NV}/cudnn/lib:"
    f"{_NV}/cufft/lib:{_NV}/curand/lib:{_NV}/cusolver/lib:"
    f"{_NV}/cusparse/lib:{_NV}/cusparselt/lib:"
    f"{_NV}/nccl/lib:{_NV}/nvtx/lib:{_NV}/nvshmem/lib:"
    f"{_NV}/nvjitlink/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
)

NVIDIA_PACKAGES = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvtx-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvshmem-cu12",
    "triton",
]

# Tokenizer artifact URLs (Karpathy's published tokenizer used by speedrun setups)
TOKENIZER_PKL_URL = (
    "https://huggingface.co/karpathy/nanochat-d32/resolve/main/tokenizer.pkl"
)
TOKEN_BYTES_URL = (
    "https://huggingface.co/karpathy/nanochat-d32/resolve/main/token_bytes.pt"
)

# --- Image: bake heavy setup at build time ---
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils")
    .pip_install("uv", "transformers", "tiktoken", *NVIDIA_PACKAGES)
    .run_commands(
        "git clone --depth 1 https://github.com/karpathy/nanochat.git /root/nanochat",
        "cd /root/nanochat && uv sync --inexact --python /usr/local/bin/python3.11",
        "find /root/nanochat -name 'speedrun.sh' | head -1 | xargs -I{} cp {} /root/nanochat/speedrun.sh",
        "chmod +x /root/nanochat/speedrun.sh",
        "sed -i '1 a export WANDB_MODE=disabled' /root/nanochat/speedrun.sh",
        # Keep a copy in the image cache too (not strictly required once volume copy exists)
        "mkdir -p /root/.cache/nanochat/tokenizer",
        f"curl -L -o /root/.cache/nanochat/tokenizer/tokenizer.pkl {TOKENIZER_PKL_URL}",
        f"curl -L -o /root/.cache/nanochat/tokenizer/token_bytes.pt {TOKEN_BYTES_URL}",
    )
    .env(
        {
            "WANDB_MODE": "disabled",
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            "PYTHONPATH": "/root/nanochat",
            "TOKENIZERS_PARALLELISM": "false",
            "UV_PYTHON": "/usr/local/bin/python3.11",
            # Crucial: tells nanochat where to look for base_data/ and tokenizer/
            "NANOCHAT_BASE_DIR": NANOCHAT_BASE_DIR,
        }
    )
)

REPO_DIR = Path("/root/nanochat")
VENV_PYTHON = "/root/nanochat/.venv/bin/python"


def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command, stream stdout+stderr in real time, print heartbeat if silent."""
    base_env = os.environ.copy()
    if env:
        base_env.update(env)

    print(f"\n[EXEC] {cmd}\n", flush=True)

    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=cwd,
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_output_time = time.time()
    start_time = time.time()

    def _heartbeat():
        while proc.poll() is None:
            time.sleep(30)
            silence = int(time.time() - last_output_time)
            elapsed = int(time.time() - start_time)
            if silence >= 30:
                print(
                    f"[HEARTBEAT] Still running... "
                    f"(silent for {silence}s, total elapsed {elapsed}s)",
                    flush=True,
                )

    threading.Thread(target=_heartbeat, daemon=True).start()

    for line in proc.stdout:
        last_output_time = time.time()
        print(line, end="", flush=True)

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


@app.function(
    image=image,
    volumes={str(VOL_PATH): vol},
    timeout=30 * 60,
    # CPU-only
)
def ensure_tokenizer_on_volume():
    """
    Make sure /vol/tokenizer/tokenizer.pkl and token_bytes.pt exist.

    With NANOCHAT_BASE_DIR=/vol, nanochat will try to load tokenizer from /vol/tokenizer [web:68].
    """
    tok_dir = VOL_PATH / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = tok_dir / "tokenizer.pkl"
    bytes_path = tok_dir / "token_bytes.pt"

    if pkl_path.exists() and bytes_path.exists():
        print(f"Tokenizer already present in {tok_dir}, skipping.")
        return f"Tokenizer already present in {tok_dir}"

    print(f"Downloading tokenizer artifacts into {tok_dir}...")
    _run(f"curl -L -o '{pkl_path}' {TOKENIZER_PKL_URL}")
    _run(f"curl -L -o '{bytes_path}' {TOKEN_BYTES_URL}")

    vol.commit()
    return f"Tokenizer saved to {tok_dir}"


@app.function(
    image=image,
    volumes={str(VOL_PATH): vol},
    timeout=2 * 60 * 60,
    # CPU-only
)
def download_dataset(num_shards: int):
    """
    Download parquet shards into the persistent Modal Volume.

    With NANOCHAT_BASE_DIR=/vol, nanochat writes to /vol/base_data by default [web:68].
    """
    target_dir = VOL_PATH / "base_data"
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = list(target_dir.glob("shard_*.parquet"))
    if len(existing) >= num_shards:
        print(f"Dataset already has {len(existing)} shards in {target_dir}, skipping.")
        return f"Dataset already has {len(existing)} shards"

    print(f"Downloading {num_shards} dataset shards using 32 parallel workers...")
    _run(
        f"uv run --no-sync python -m nanochat.dataset -n {num_shards} -w 32",
        cwd=REPO_DIR,
        env={"NANOCHAT_BASE_DIR": NANOCHAT_BASE_DIR},
    )

    vol.commit()
    return f"Downloaded {num_shards} shards into {target_dir}"


@app.function(
    image=image,
    gpu="H100:8",
    timeout=24 * 60 * 60,
    volumes={str(VOL_PATH): vol},
    enable_memory_snapshot=True,
)
def run_speedrun(model: str = "d12", force_restart: bool = False):
    if force_restart:
        _run(f"rm -rf '{RUNS_DIR}/{model}'")

    print(f"Starting training for {model} on 8xH100...", flush=True)

    _run(
        "./speedrun.sh",
        cwd=REPO_DIR,
        env={
            "MODEL": model,
            "PYTHONUNBUFFERED": "1",
            "NANOCHAT_BASE_DIR": NANOCHAT_BASE_DIR,
        },
    )

    vol.commit()
    return f"Done. Outputs in {RUNS_DIR}"


@app.function(
    image=image,
    gpu="H100:1",
    timeout=30 * 60,
    volumes={str(VOL_PATH): vol},
    enable_memory_snapshot=True,
)
def smoke_test_10_steps(model: str = "d12"):
    print("Starting single-GPU smoke test...", flush=True)
    print("Heartbeat will print every 30s during silent torch.compile phases.", flush=True)

    cmd = (
        f"{VENV_PYTHON} scripts/base_train.py --run=d12_test "
        "--num-iterations=10 --core-metric-every=1 "
        "--eval-every=5 --save-every=5 --device-batch-size=16"
    )

    _run(
        cmd,
        cwd=REPO_DIR,
        env={
            "PYTHONUNBUFFERED": "1",
            "TORCH_LOGS": "+inductor,+dynamo",
            "NANOCHAT_BASE_DIR": NANOCHAT_BASE_DIR,
        },
    )

    vol.commit()
    return "Smoke test complete."


@app.local_entrypoint()
def main(task: str = "run", model: str = "d12"):
    if task == "test":
        # Ensure tokenizer + 1 shard exist on the shared volume, then run the smoke test
        ensure_tokenizer_on_volume.remote()
        download_dataset.remote(num_shards=1)
        print(smoke_test_10_steps.remote(model=model))
    else:
        # Ensure tokenizer + full dataset exist on the shared volume, then run the full speedrun
        ensure_tokenizer_on_volume.remote()
        download_dataset.remote(num_shards=240)
        print(run_speedrun.remote(model=model))
