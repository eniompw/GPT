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
CACHE_DIR = VOL_PATH / "cache"
LOCAL_DATA_DIR = "/tmp/nanochat_local"
REPO_DIR = Path("/root/nanochat")
VENV_PYTHON = "/root/nanochat/.venv/bin/python"

_NV = "/usr/local/lib/python3.11/site-packages/nvidia"
_NV_LIBS = [
    "cuda_runtime", "cuda_cupti", "cublas", "cudnn", "cufft", "curand",
    "cusolver", "cusparse", "cusparselt", "nccl", "nvtx", "nvshmem", "nvjitlink",
]
LD_LIBRARY_PATH = ":".join(
    [f"{_NV}/{lib}/lib" for lib in _NV_LIBS]
    + ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
)

NVIDIA_PACKAGES = [
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",       "nvidia-cudnn-cu12",       "nvidia-cufft-cu12",
    "nvidia-curand-cu12",       "nvidia-cusolver-cu12",    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",   "nvidia-nccl-cu12",        "nvidia-nvtx-cu12",
    "nvidia-nvjitlink-cu12",    "nvidia-nvshmem-cu12",     "triton",
]

TOKENIZER_PKL_URL = "https://huggingface.co/karpathy/nanochat-d32/resolve/main/tokenizer.pkl"
TOKEN_BYTES_URL   = "https://huggingface.co/karpathy/nanochat-d32/resolve/main/token_bytes.pt"
EVAL_BUNDLE_URL   = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "ARROW_USER_THREADS_DEFAULT": "1",
    "PYARROW_NUM_THREADS": "1",
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils", "unzip", "rsync")
    .pip_install("uv", "transformers", "tiktoken", *NVIDIA_PACKAGES)
    .run_commands(
        "git clone --depth 1 https://github.com/karpathy/nanochat.git /root/nanochat",
        "cd /root/nanochat && uv sync --inexact --python /usr/local/bin/python3.11",
        "find /root/nanochat -name 'speedrun.sh' | head -1 | xargs -I{} cp {} /root/nanochat/speedrun.sh",
        "chmod +x /root/nanochat/speedrun.sh",
        "sed -i '1 a export WANDB_MODE=disabled' /root/nanochat/speedrun.sh",
        "sed -i 's/base_train -- /base_train -- --save-every=500 --eval-every=200 /' /root/nanochat/speedrun.sh",
        "mkdir -p /root/.cache/nanochat/tokenizer",
    )
    .env({
        "WANDB_MODE":                    "disabled",
        "LD_LIBRARY_PATH":               LD_LIBRARY_PATH,
        "PYTHONPATH":                    "/root/nanochat",
        "TOKENIZERS_PARALLELISM":        "false",
        "UV_PYTHON":                     "/usr/local/bin/python3.11",
        "NANOCHAT_BASE_DIR":             str(VOL_PATH),
        "TORCHINDUCTOR_CACHE_DIR":       "/tmp/torch_cache",
        "TRITON_CACHE_DIR":              "/tmp/triton_cache",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_LOGS":                    "+dynamo,+inductor",
        "TORCHINDUCTOR_VERBOSE":         "1",
        "PYTHONFAULTHANDLER":            "1",
    })
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    base_env = {**os.environ, **(env or {})}
    print(f"\n[EXEC] {cmd}\n", flush=True)
    proc = subprocess.Popen(
        ["stdbuf", "-oL", "bash", "-c", cmd],
        cwd=cwd, env=base_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    last_out = [time.time()]
    start = time.time()
    cache_synced = [False]

    def _heartbeat():
        while proc.poll() is None:
            time.sleep(60)
            silence = int(time.time() - last_out[0])
            if silence >= 60:
                print(f"[HEARTBEAT] silent {silence}s / total {int(time.time()-start)}s", flush=True)

    threading.Thread(target=_heartbeat, daemon=True).start()
    for line in proc.stdout:
        last_out[0] = time.time()
        print(line, end="", flush=True)
        if not cache_synced[0] and "Step 00001" in line:
            print("[INFO] All compilations done, syncing caches to volume...", flush=True)
            _sync_caches_out(vol)
            cache_synced[0] = True
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _sync_caches_in() -> None:
    _run(f"mkdir -p /tmp/torch_cache /tmp/triton_cache {CACHE_DIR}/torch_cache {CACHE_DIR}/triton_cache")
    _run(f"cp -a {CACHE_DIR}/torch_cache/.  /tmp/torch_cache/  2>/dev/null || true")
    _run(f"cp -a {CACHE_DIR}/triton_cache/. /tmp/triton_cache/ 2>/dev/null || true")


def _sync_caches_out(vol_ref: modal.Volume) -> None:
    _run(f"cp -a /tmp/torch_cache/.  {CACHE_DIR}/torch_cache/")
    _run(f"cp -a /tmp/triton_cache/. {CACHE_DIR}/triton_cache/")
    vol_ref.commit()


def _sync_data_in() -> None:
    _run(f"mkdir -p {LOCAL_DATA_DIR}/{{base_data,tokenizer,runs,eval_bundle}}")
    _run(f"rsync -a --info=progress2 {VOL_PATH}/base_data/ {LOCAL_DATA_DIR}/base_data/")
    _run(f"cp -a {VOL_PATH}/tokenizer/. {LOCAL_DATA_DIR}/tokenizer/")
    _run(f"cp -a {VOL_PATH}/eval_bundle/. {LOCAL_DATA_DIR}/eval_bundle/ 2>/dev/null || true")
    _run(f"cp -a {RUNS_DIR}/.             {LOCAL_DATA_DIR}/runs/         2>/dev/null || true")


def _sync_runs_out(vol_ref: modal.Volume) -> None:
    _run(f"mkdir -p {RUNS_DIR}")
    _run(f"cp -a {LOCAL_DATA_DIR}/runs/. {RUNS_DIR}/")
    vol_ref.commit()


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOL_PATH): vol}, timeout=30 * 60)
def ensure_tokenizer_on_volume():
    tok_dir = VOL_PATH / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = tok_dir / "tokenizer.pkl"
    bytes_path = tok_dir / "token_bytes.pt"
    if not (pkl_path.exists() and bytes_path.exists()):
        _run(f"curl -L -o '{pkl_path}'   {TOKENIZER_PKL_URL}")
        _run(f"curl -L -o '{bytes_path}' {TOKEN_BYTES_URL}")
    if not (VOL_PATH / "eval_bundle").exists():
        _run(f"curl -L -o /tmp/eval_bundle.zip {EVAL_BUNDLE_URL}")
        _run(f"unzip -q /tmp/eval_bundle.zip -d {VOL_PATH}")
        _run("rm /tmp/eval_bundle.zip")
    vol.commit()


@app.function(image=image, volumes={str(VOL_PATH): vol}, timeout=2 * 60 * 60)
def download_dataset(num_shards: int):
    target_dir = VOL_PATH / "base_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    if len(list(target_dir.glob("shard_*.parquet"))) >= num_shards:
        return
    _run(
        f"uv run --no-sync python -m nanochat.dataset -n {num_shards} -w 32",
        cwd=REPO_DIR,
        env={"NANOCHAT_BASE_DIR": str(VOL_PATH)},
    )
    vol.commit()


@app.function(image=image, gpu="H100:8", timeout=24 * 60 * 60, volumes={str(VOL_PATH): vol})
def run_speedrun(model: str = "d12", force_restart: bool = False):
    if force_restart:
        _run(f"rm -rf '{RUNS_DIR}/{model}'")
    _sync_caches_in()
    _sync_data_in()
    _run(
        "./speedrun.sh",
        cwd=REPO_DIR,
        env={"MODEL": model, "PYTHONUNBUFFERED": "1",
             "NANOCHAT_BASE_DIR": LOCAL_DATA_DIR, **_THREAD_ENV},
    )
    _sync_runs_out(vol)


@app.function(image=image, gpu="H100:1", timeout=30 * 60, volumes={str(VOL_PATH): vol})
def smoke_test_10_steps(model: str = "d12"):
    _sync_caches_in()
    _sync_data_in()
    _run(
        f"{VENV_PYTHON} -u scripts/base_train.py --run=d12_test "
        "--num-iterations=10 --device-batch-size=4 "
        "--total-batch-size=8192 "
        "--eval-every=-1 --core-metric-every=-1 --sample-every=-1",
        cwd=REPO_DIR,
        env={
            "PYTHONUNBUFFERED":    "1",
            "NANOCHAT_BASE_DIR":   LOCAL_DATA_DIR,
            "TORCHDYNAMO_DISABLE": "1",
            **_THREAD_ENV,
        },
    )
    _sync_caches_out(vol)


@app.local_entrypoint()
def main(task: str = "run", model: str = "d12", force_restart: bool = False):
    ensure_tokenizer_on_volume.remote()
    num_shards = 2 if task == "test" else 240
    download_dataset.remote(num_shards=num_shards)
    if task == "test":
        smoke_test_10_steps.remote(model=model)
    else:
        run_speedrun.remote(model=model, force_restart=force_restart)
