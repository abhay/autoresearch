"""
CUDA regression test on Modal.

Usage:
    pip install modal   # one-time
    modal setup         # one-time, creates account + API token
    modal run test_modal.py
"""

import modal

app = modal.App("autoresearch-test")

# Only copy the files needed to build and run
_PROJECT_FILES = [
    "backend.py",
    "prepare.py",
    "train.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
]

def _copy_project_files(image):
    for f in _PROJECT_FILES:
        image = image.add_local_file(f, f"/app/{f}", copy=True)
    return image

# Build image with uv + project dependencies pre-installed
image = _copy_project_files(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
).run_commands("cd /app && uv sync", gpu="H100")


@app.function(gpu="H100", timeout=900, image=image)
def test_training():
    import subprocess
    import os
    os.chdir("/app")

    # Download data (just 2 shards — enough for a regression test)
    print("=== Preparing data ===")
    r = subprocess.run(
        ["uv", "run", "prepare.py", "--num-shards", "2"],
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError("prepare.py failed")

    # Run training
    print("\n=== Running training ===")
    r = subprocess.run(
        ["uv", "run", "train.py"],
        capture_output=True, text=True,
        timeout=600,
    )
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr)
        raise RuntimeError("train.py failed")

    # Extract and return key metrics
    for line in r.stdout.splitlines():
        if line.startswith(("val_bpb:", "training_seconds:", "peak_vram_mb:", "mfu_percent:", "num_steps:", "depth:")):
            print(line)


@app.local_entrypoint()
def main():
    test_training.remote()
