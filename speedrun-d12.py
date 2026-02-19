import os
import re
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

_NV = "/usr/local/lib/python3.11/site-packages/nvidia"
LD_LIBRARY_PATH = ":".join([
    f"{_NV}/cuda_runtime/lib", f"{_NV}/cuda_cupti/lib",
    f"{_NV}/cublas/lib",       f"{_NV}/cudnn/lib",
    f"{_NV}/cufft/lib",        f"{_NV}/curand/lib",  f"{_NV}/cusolver/lib",
    f"{_NV}/cusparse/lib",     f"{_NV}/cusparselt/lib",
    f"{_NV}/nccl/lib",         f"{_NV}/nvtx/lib",    f"{_NV}/nvshmem/lib",
    f"{_NV}/nvjitlink/lib",    "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu",
])

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

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils", "unzip")
    .pip_install("uv", "transformers", "tiktoken", *NVIDIA_PACKAGES)
    .run_commands(
        "git clone --depth 1 https://github.com/karpathy/nanochat.git /root/nanochat",
        "cd /root/nanochat && uv sync --inexact --python /usr/local/bin/python3.11",
        "find /root/nanochat -name 'speedrun.sh' | head -1 | xargs -I{} cp {} /root/nanochat/speedrun.sh",
        "chmod +x /root/nanochat/speedrun.sh",
        "sed -i '1 a export WANDB_MODE=disabled' /root/nanochat/speedrun.sh",
        "mkdir -p /root/.cache/nanochat/tokenizer",
    )
    .env({
        "WANDB_MODE":                  "disabled",
        "LD_LIBRARY_PATH":             LD_LIBRARY_PATH,
        "PYTHONPATH":                  "/root/nanochat",
        "TOKENIZERS_PARALLELISM":      "false",
        "UV_PYTHON":                   "/usr/local/bin/python3.11",
        "NANOCHAT_BASE_DIR":           str(VOL_PATH),
        "TORCHINDUCTOR_CACHE_DIR":     "/tmp/torch_cache",
        "TRITON_CACHE_DIR":            "/tmp/triton_cache",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTHONFAULTHANDLER":          "1",
    })
)

REPO_DIR    = Path("/root/nanochat")
VENV_PYTHON = "/root/nanochat/.venv/bin/python"

# Shared: keep CPU thread pools tame on every training invocation
_THREAD_ENV = {
    "OMP_NUM_THREADS":           "1",
    "MKL_NUM_THREADS":           "1",
    "ARROW_USER_THREADS_DEFAULT": "1",
    "PYARROW_NUM_THREADS":       "1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    base_env = {**os.environ, **(env or {})}
    print(f"\n[EXEC] {cmd}\n", flush=True)
    proc = subprocess.Popen(
        ["stdbuf", "-oL", "bash", "-lc", cmd],
        cwd=cwd, env=base_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    last_out = [time.time()]
    start    = time.time()

    def _heartbeat():
        while proc.poll() is None:
            time.sleep(30)
            silence = int(time.time() - last_out[0])
            if silence >= 30:
                print(f"[HEARTBEAT] silent {silence}s / total {int(time.time()-start)}s", flush=True)

    threading.Thread(target=_heartbeat, daemon=True).start()
    for line in proc.stdout:
        last_out[0] = time.time()
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _sync_caches_in() -> None:
    # Create dirs for both sides in one shot, then copy vol→tmp
    _run(f"mkdir -p /tmp/torch_cache /tmp/triton_cache {CACHE_DIR}/torch_cache {CACHE_DIR}/triton_cache")
    _run(f"cp -a {CACHE_DIR}/torch_cache/.  /tmp/torch_cache/  2>/dev/null || true")
    _run(f"cp -a {CACHE_DIR}/triton_cache/. /tmp/triton_cache/ 2>/dev/null || true")


def _sync_caches_out(vol_ref: modal.Volume) -> None:
    _run(f"cp -a /tmp/torch_cache/.  {CACHE_DIR}/torch_cache/")
    _run(f"cp -a /tmp/triton_cache/. {CACHE_DIR}/triton_cache/")
    vol_ref.commit()


def _sync_data_in() -> None:
    print("Copying dataset + tokenizer /vol → /tmp...", flush=True)
    # Brace expansion creates all four subdirs in a single shell call
    _run(f"mkdir -p {LOCAL_DATA_DIR}/{{base_data,tokenizer,runs,eval_bundle}}")
    _run(f"cp -a {VOL_PATH}/base_data/. {LOCAL_DATA_DIR}/base_data/")
    _run(f"cp -a {VOL_PATH}/tokenizer/. {LOCAL_DATA_DIR}/tokenizer/")
    _run(f"cp -a {VOL_PATH}/eval_bundle/. {LOCAL_DATA_DIR}/eval_bundle/ 2>/dev/null || true")
    _run(f"cp -a {RUNS_DIR}/.            {LOCAL_DATA_DIR}/runs/         2>/dev/null || true")


def _sync_runs_out(vol_ref: modal.Volume) -> None:
    _run(f"mkdir -p {RUNS_DIR}")
    _run(f"cp -a {LOCAL_DATA_DIR}/runs/. {RUNS_DIR}/")
    vol_ref.commit()


def _patch_train_script() -> None:
    """
    Patch base_train.py and dataloader.py directly in Python.
    No bash heredoc needed — this function runs inside the Modal container.
    """
    train = REPO_DIR / "scripts/base_train.py"
    lines = train.read_text().splitlines(True)

    if "faulthandler.dump_traceback_later" not in "".join(lines):
        lines.insert(0, "import faulthandler; faulthandler.dump_traceback_later(60, repeat=True)\n")

    injections = [
        ("compile", "torch.compile",  "[DEBUG] Reached torch.compile block..."),
        ("loader",  "train_loader =", "[DEBUG] Reached DataLoader initialization..."),
        ("loop",    "while step <",   "[DEBUG] Entering main training loop!"),
    ]
    seen, out = set(), []
    for line in lines:
        indent = line[: len(line) - len(line.lstrip())]
        for key, trigger, msg in injections:
            if key not in seen and trigger in line:
                out.append(f'{indent}print("{msg}", flush=True)\n')
                seen.add(key)
        out.append(line)
    train.write_text("".join(out))

    # Cap ProcessPool workers in the dataloader (smoke-test stability)
    dlp = REPO_DIR / "nanochat/dataloader.py"
    dl  = dlp.read_text()
    dl2 = re.sub(r"max_workers\s*=\s*\d+",              "max_workers=1", dl)
    dl2 = re.sub(r"max_workers\s*=\s*os\.cpu_count\(\)", "max_workers=1", dl2)
    if dl2 != dl:
        dlp.write_text(dl2)

    print("Patched base_train.py and dataloader.py", flush=True)


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={str(VOL_PATH): vol}, timeout=30 * 60)
def ensure_tokenizer_on_volume():
    tok_dir = VOL_PATH / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    pkl_path   = tok_dir / "tokenizer.pkl"
    bytes_path = tok_dir / "token_bytes.pt"
    if not (pkl_path.exists() and bytes_path.exists()):
        _run(f"curl -L -o '{pkl_path}'   {TOKENIZER_PKL_URL}")
        _run(f"curl -L -o '{bytes_path}' {TOKEN_BYTES_URL}")
    if not (VOL_PATH / "eval_bundle").exists():
        _run(f"curl -L -o /tmp/eval_bundle.zip {EVAL_BUNDLE_URL}")
        _run(f"unzip -q /tmp/eval_bundle.zip -d {VOL_PATH}")
        _run("rm /tmp/eval_bundle.zip")
    vol.commit()
    return f"Tokenizer and eval_bundle saved to {VOL_PATH}"


@app.function(image=image, volumes={str(VOL_PATH): vol}, timeout=2 * 60 * 60)
def download_dataset(num_shards: int):
    target_dir = VOL_PATH / "base_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob("shard_*.parquet"))
    if len(existing) >= num_shards:
        return f"Dataset already has {len(existing)} shards, skipping."
    _run(
        f"uv run --no-sync python -m nanochat.dataset -n {num_shards} -w 32",
        cwd=REPO_DIR,
        env={"NANOCHAT_BASE_DIR": str(VOL_PATH)},
    )
    vol.commit()
    return f"Downloaded {num_shards} shards into {target_dir}"


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
    _sync_caches_out(vol)
    _sync_runs_out(vol)
    return f"Done. Outputs in {RUNS_DIR}"


@app.function(image=image, gpu="H100:1", timeout=30 * 60, volumes={str(VOL_PATH): vol})
def smoke_test_10_steps(model: str = "d12"):
    print("Starting smoke test...", flush=True)
    _sync_caches_in()
    _sync_data_in()
    _patch_train_script()  # pure Python — no bash heredoc needed
    _run(
        f"{VENV_PYTHON} -u scripts/base_train.py --run=d12_test "
        "--num-iterations=10 --device-batch-size=16",
        cwd=REPO_DIR,
        env={
            "PYTHONUNBUFFERED":          "1",
            "NANOCHAT_BASE_DIR":         LOCAL_DATA_DIR,
            "TORCHDYNAMO_DISABLE":       "1",
            "TORCH_COMPILE_DISABLE":     "1",
            "TRITON_PRINT_AUTOTUNING":   "1",
            "FLASH_ATTENTION_FORCE_DISABLE": "1",
            "TORCH_SDPA_FORCE_ENABLED":  "1",
            "MASTER_ADDR":               "127.0.0.1",
            "MASTER_PORT":               "29500",
            "RANK": "0", "WORLD_SIZE": "1",
            "NCCL_SOCKET_IFNAME":        "lo",
            "NCCL_DEBUG":                "INFO",
            **_THREAD_ENV,
        },
    )
    _sync_caches_out(vol)
    return "Smoke test complete."


@app.local_entrypoint()
def main(task: str = "run", model: str = "d12"):
    if task == "test":
        ensure_tokenizer_on_volume.remote()
        download_dataset.remote(num_shards=1)
        print(smoke_test_10_steps.remote(model=model))
    else:
        ensure_tokenizer_on_volume.remote()
        download_dataset.remote(num_shards=240)
        print(run_speedrun.remote(model=model))
