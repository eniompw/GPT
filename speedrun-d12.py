import os
import subprocess
from pathlib import Path
import modal


APP_NAME = "nanochat-speedrun-h100"
VOLUME_NAME = "nanochat-persistent-storage"
GPU_CONFIG = "H100:1"


app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


VOL_PATH = Path("/vol")
RUNS_DIR = VOL_PATH / "runs"
DATA_DIR = VOL_PATH / "data"


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
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
    "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12", "nvidia-nvtx-cu12", "nvidia-nvjitlink-cu12", "nvidia-nvshmem-cu12",
    "triton",
]


# --- Image: all heavy setup baked in at build time ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils")
    .pip_install("uv", "transformers", "tiktoken", *NVIDIA_PACKAGES)
    .run_commands(
        # Clone repo
        "git clone --depth 1 https://github.com/karpathy/nanochat.git /root/nanochat",
        # Force uv to use system python3.11
        "cd /root/nanochat && uv sync --inexact --python /usr/local/bin/python3.11",
        # Find speedrun.sh and ensure it's in the root
        "find /root/nanochat -name 'speedrun.sh' | head -1 | xargs -I{} cp {} /root/nanochat/speedrun.sh",
        "chmod +x /root/nanochat/speedrun.sh",
        # Patch wandb disable
        "sed -i '1 a export WANDB_MODE=disabled' /root/nanochat/speedrun.sh",
        # Pre-download tokenizer
        "mkdir -p /root/.cache/nanochat/tokenizer",
        "curl -L -o /root/.cache/nanochat/tokenizer/tokenizer.pkl "
            "https://huggingface.co/karpathy/nanochat-d32/resolve/main/tokenizer.pkl",
        "curl -L -o /root/.cache/nanochat/tokenizer/token_bytes.pt "
            "https://huggingface.co/karpathy/nanochat-d32/resolve/main/token_bytes.pt",
    )
    .env({
        "WANDB_MODE": "disabled",
        "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        "PYTHONPATH": "/root/nanochat",
        "TOKENIZERS_PARALLELISM": "false",
        "UV_PYTHON": "/usr/local/bin/python3.11",
    })
)


REPO_DIR = Path("/root/nanochat")


def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    print(f"\n[EXEC] {cmd}\n", flush=True)
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=base_env, check=True)


def _ensure_dataset(num_shards: int = 2) -> None:
    data_dir = Path("/root/.cache/nanochat/base_data")
    existing = list(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if len(existing) >= num_shards:
        print(f"Dataset already has {len(existing)} shards, skipping download.")
        return
    print(f"Downloading {num_shards} dataset shards...")
    _run(
        f"uv run --no-sync python -m nanochat.dataset -n {num_shards} -w 4",
        cwd=REPO_DIR,
    )


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=24 * 60 * 60,
    volumes={str(VOL_PATH): vol},
    enable_memory_snapshot=True,
)
def run_speedrun(model: str = "d12", force_restart: bool = False):
    _ensure_dataset()

    if force_restart:
        _run(f"rm -rf '{RUNS_DIR}/{model}'")

    print(f"Starting training for {model}...", flush=True)
    print("NOTE: The first few minutes may be silent due to torch.compile() shader compilation.", flush=True)

    # Added python -u for unbuffered output to see logs faster
    _run(
        f"export MODEL={model} && python -u ./speedrun.sh",
        cwd=REPO_DIR,
    )

    vol.commit()
    return f"Done. Outputs in {RUNS_DIR}"


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=30 * 60,
    volumes={str(VOL_PATH): vol},
    enable_memory_snapshot=True,
)
def smoke_test_10_steps(model: str = "d12"):
    _ensure_dataset(num_shards=1)
    
    print("Starting smoke test...", flush=True)
    print("NOTE: The first 2-3 minutes will be silent while the H100 compiles kernels (torch.compile).", flush=True)

    if (REPO_DIR / "scripts" / "base_train.py").exists():
        cmd = (
            "uv run --no-sync python -u scripts/base_train.py --run=d12_test "
            "--num-iterations=10 --core-metric-every=1 "
            "--eval-every=5 --save-every=5 --device-batch-size=16"
        )
    else:
        cmd = (
            f"uv run --no-sync python -u -m nanochat.train config/{model}.py "
            "--max_iters=10 --log_interval=1 "
            "--eval_interval=5 --always_save_checkpoint=True"
        )

    _run(cmd, cwd=REPO_DIR)

    vol.commit()
    return "Smoke test complete."


@app.local_entrypoint()
def main(task: str = "run", model: str = "d12"):
    if task == "test":
        print(smoke_test_10_steps.remote(model=model))
    else:
        print(run_speedrun.remote(model=model))
