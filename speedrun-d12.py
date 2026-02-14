import os
import subprocess
from pathlib import Path
import modal

APP_NAME = "nanochat-speedrun-h100"
VOLUME_NAME = "nanochat-persistent-storage"
GPU_CONFIG = "H100:8"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_PATH = Path("/vol")
RUNS_DIR = VOL_PATH / "runs"
DATA_DIR = VOL_PATH / "data"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget", "build-essential", "pkg-config", "findutils")
    .pip_install("uv", "transformers", "tiktoken") # Added explicit deps just in case
    .env({"WANDB_MODE": "disabled"})
)

def _run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> None:
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    print(f"\n[EXEC] {cmd}\n", flush=True)
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=base_env, check=True)

def _find_file(name: str, root: Path) -> Path | None:
    try:
        cmd = ["find", str(root), "-name", name, "-not", "-path", "*/.*/*"]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        if out:
            return Path(sorted(out, key=len)[0])
    except Exception:
        return None
    return None

def _setup_repo(repo_ref: str, repo_url: str) -> Path:
    workdir = Path("/root/nanochat_work")
    repo_dir = workdir / "nanochat"

    _run(f"mkdir -p '{RUNS_DIR}' '{DATA_DIR}'")
    if not repo_dir.exists():
        _run(f"mkdir -p '{workdir}'")
        _run(f"git clone --depth 1 '{repo_url}' '{repo_dir}'")

    _run("git fetch --all --tags --prune", cwd=repo_dir)
    _run(f"git checkout '{repo_ref}'", cwd=repo_dir)

    _run(f"rm -rf data && ln -s '{DATA_DIR}' data", cwd=repo_dir)
    _run(f"rm -rf logs && ln -s '{RUNS_DIR}' logs", cwd=repo_dir)

    sr = _find_file("speedrun.sh", repo_dir)
    if sr and sr.exists():
        _run(f"chmod +x '{sr}'", cwd=repo_dir)
        if sr.parent.name == "runs" and not (repo_dir / "speedrun.sh").exists():
            _run("cp runs/speedrun.sh ./speedrun.sh && chmod +x ./speedrun.sh", cwd=repo_dir)

    return repo_dir

def _ensure_uv_env_has_cuda_bits(repo_dir: Path) -> None:
    _run("uv sync --inexact", cwd=repo_dir)

    _run(
        f"uv run python -c \"import site, pathlib; "
        f"p = pathlib.Path(site.getsitepackages()[0]) / 'nanochat_repo.pth'; "
        f"p.write_text('{repo_dir}\\n')\"",
        cwd=repo_dir,
    )

    print("Installing NVIDIA library wheels for PyTorch...")
    packages = [
        "nvidia-cuda-runtime-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
        "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
        "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12", "nvidia-nvtx-cu12", "nvidia-nvjitlink-cu12", "nvidia-nvshmem-cu12",
        "triton",
    ]
    _run(f"uv pip install {' '.join(packages)}", cwd=repo_dir)

def _uv_run(repo_dir: Path, cmd: str) -> None:
    helper_script_path = repo_dir / "find_nvidia_libs.py"
    if not helper_script_path.exists():
        with open(helper_script_path, "w") as f:
            f.write("""
import os
import site

def find_libs():
    libs = []
    paths = site.getsitepackages() + [site.getusersitepackages()]
    for p in paths:
        if not p:
            continue
        nv_path = os.path.join(p, 'nvidia')
        if os.path.exists(nv_path):
            for d in os.listdir(nv_path):
                lib_path = os.path.join(nv_path, d, 'lib')
                if os.path.exists(lib_path):
                    libs.append(lib_path)
    print(':'.join(list(set(libs))))

if __name__ == '__main__':
    find_libs()
""")

    try:
        lib_paths = subprocess.check_output(
            ["bash", "-lc", f"uv run python {helper_script_path}"], cwd=repo_dir, text=True
        ).strip()

        sys_cuda = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
        # Disable tokenizers parallelism to avoid deadlocks in forks
        full_cmd = f"export TOKENIZERS_PARALLELISM=false && export WANDB_MODE=disabled && export LD_LIBRARY_PATH={lib_paths}:{sys_cuda}:$LD_LIBRARY_PATH && export PYTHONPATH={repo_dir}:$PYTHONPATH && uv run --no-sync {cmd}"
        _run(full_cmd, cwd=repo_dir)
    except subprocess.CalledProcessError:
        _run(f"export WANDB_MODE=disabled && export PYTHONPATH={repo_dir}:$PYTHONPATH && uv run {cmd}", cwd=repo_dir)

def _patch_speedrun_for_checkpoints(repo_dir: Path, eval_interval: int = 100) -> None:
    sr = repo_dir / "speedrun.sh"
    if not sr.exists(): return

    flags = f" --eval-every={eval_interval} --save-every={eval_interval}"

    _run("sed -i '1 a export WANDB_MODE=disabled' speedrun.sh", cwd=repo_dir)

    for needle in ["scripts.base_train", "scripts.mid_train", "scripts.chat_sft", "python -m", "uv run"]:
        _run(f"grep -q '{needle}' speedrun.sh && sed -i '/{needle}/ s/$/{flags}/' speedrun.sh || true", cwd=repo_dir)

def _ensure_tokenizer(repo_dir: Path) -> None:
    print("Ensuring tokenizer is present...")
    tokenizer_dir = Path("/root/.cache/nanochat/tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "tokenizer.pkl": "https://huggingface.co/karpathy/nanochat-d32/resolve/main/tokenizer.pkl",
        "token_bytes.pt": "https://huggingface.co/karpathy/nanochat-d32/resolve/main/token_bytes.pt"
    }

    for filename, url in files.items():
        filepath = tokenizer_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            subprocess.run(["curl", "-L", "-o", str(filepath), url], check=True)
        else:
            print(f"{filename} already exists.")

def _ensure_dataset(repo_dir: Path, num_shards: int = 2) -> None:
    print("Ensuring dataset is present...")
    # nanochat expects parquet shards in ~/.cache/nanochat/base_data/
    # named shard_XXXXX.parquet. The last shard is used for validation,
    # so we need at least 2. Run the built-in dataset download script.
    data_dir = Path("/root/.cache/nanochat/base_data")
    existing = list(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if len(existing) >= num_shards:
        print(f"Dataset already has {len(existing)} shards, skipping download.")
        return
    print(f"Downloading {num_shards} dataset shards via nanochat.dataset...")
    _uv_run(repo_dir, f"python -m nanochat.dataset -n {num_shards} -w 4")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=24 * 60 * 60,
    volumes={str(VOL_PATH): vol},
)
def run_speedrun(repo_ref: str = "master", model: str = "d12", force_restart: bool = False):
    repo_dir = _setup_repo(repo_ref=repo_ref, repo_url="https://github.com/karpathy/nanochat.git")
    _ensure_uv_env_has_cuda_bits(repo_dir)
    _patch_speedrun_for_checkpoints(repo_dir, eval_interval=100)
    _ensure_tokenizer(repo_dir)
    _ensure_dataset(repo_dir)

    if force_restart:
        _run(f"rm -rf '{RUNS_DIR}/{model}'", cwd=repo_dir)

    helper_script_path = repo_dir / "find_nvidia_libs.py"
    if not helper_script_path.exists():
         with open(helper_script_path, "w") as f:
            f.write("import os, site; paths = site.getsitepackages() + [site.getusersitepackages()]; print(':'.join([os.path.join(p, 'nvidia', d, 'lib') for p in paths if p and os.path.exists(os.path.join(p, 'nvidia')) for d in os.listdir(os.path.join(p, 'nvidia')) if os.path.exists(os.path.join(p, 'nvidia', d, 'lib'))]))")

    try:
        lib_paths = subprocess.check_output(["bash", "-lc", f"uv run python {helper_script_path}"], cwd=repo_dir, text=True).strip()
        sys_cuda = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
        _run(f"export WANDB_MODE=disabled && export LD_LIBRARY_PATH={lib_paths}:{sys_cuda}:$LD_LIBRARY_PATH && export PYTHONPATH={repo_dir}:$PYTHONPATH && ./speedrun.sh {model}", cwd=repo_dir)
    except:
        _run(f"export WANDB_MODE=disabled && export PYTHONPATH={repo_dir}:$PYTHONPATH && ./speedrun.sh {model}", cwd=repo_dir)

    vol.commit()
    return f"Done. Outputs in {RUNS_DIR}"

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=30 * 60,
    volumes={str(VOL_PATH): vol},
)
def smoke_test_10_steps(repo_ref: str = "master", model: str = "d12"):
    repo_dir = _setup_repo(repo_ref=repo_ref, repo_url="https://github.com/karpathy/nanochat.git")
    _ensure_uv_env_has_cuda_bits(repo_dir)
    _ensure_tokenizer(repo_dir)
    _ensure_dataset(repo_dir)

    if (repo_dir / "scripts" / "base_train.py").exists():
        cmd = f"python scripts/base_train.py --run=d12_test --num-iterations=10 --core-metric-every=1 --eval-every=5 --save-every=5"
    else:
        script_path = _find_file("train.py", repo_dir)
        if script_path:
            rel = script_path.relative_to(repo_dir)
            cmd = f"python {rel} config/{model}.py --max_iters=10 --log_interval=1 --eval_interval=5 --always_save_checkpoint=True"
        else:
            cmd = f"python -m nanochat.train config/{model}.py --max_iters=10 --log_interval=1 --eval_interval=5 --always_save_checkpoint=True"

    _uv_run(repo_dir, cmd)

    vol.commit()
    return "Smoke test complete."

@app.local_entrypoint()
def main(task: str = "run", repo_ref: str = "master", model: str = "d12"):
    if task == "test":
        print(smoke_test_10_steps.remote(repo_ref=repo_ref, model=model))
    else:
        print(run_speedrun.remote(repo_ref=repo_ref, model=model))