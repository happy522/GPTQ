import modal

app = modal.App("gptq-quantization")

# -----------------------
# HF cache volume
# -----------------------
hf_cache_volume = modal.Volume.from_name(
    "hf-model-cache",
    create_if_missing=True
)

# -----------------------
# Container image
# -----------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "python3-dev"
    )
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.2","datasets==2.14.6"
"pyarrow==14.0.2",
        "accelerate==0.26.1",
        "huggingface_hub==0.20.3",
        "sentencepiece",
        "numpy",
        "scipy",
        "ninja",
    )
    .run_commands(
            "git clone https://github.com/IST-DASLab/gptq.git /root/gptq",
            "sed -i 's/allenai--c4/en/g' /root/gptq/datautils.py",

    )
    .env({"HF_HOME": "/hf-vol"})
)

# -----------------------
# GPTQ function
# -----------------------
@app.function(
    gpu="A10G",
    timeout=60 * 60,
    image=image,
    volumes={"/hf-vol": hf_cache_volume},
)
def run_gptq(
    model_name: str = "facebook/opt-125m",
    dataset: str = "wikitext2",
    wbits: int = 4,
    groupsize: int = 128,
    save_path: str = "",
):
    import subprocess
    import os

    workdir = "/root/gptq"

    # Build command safely (NO shell string bugs)
    cmd = [
        "python",
        "opt.py",
        model_name,
        dataset,
        "--wbits", str(wbits),
        "--groupsize", str(groupsize),
    ]

    if save_path:
        cmd += ["--save", save_path]

    print("Running command:", " ".join(cmd), flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream logs live
    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"GPTQ failed with exit code {process.returncode}"
        )

    return "success"


# -----------------------
# Local entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    run_gptq.remote(
        model_name="facebook/opt-125m",
        dataset="wikitext2",   # safe default (avoids C4 crash)
        wbits=4,
        groupsize=128,
    )