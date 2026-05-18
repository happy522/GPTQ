import modal

# -------------------------
# Image build
# -------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
    )
    .run_commands(
        "git clone https://github.com/happy522/GPTQ /workspace/gptq"
    )
)

app = modal.App("gptq-runner")


# -------------------------
# GPTQ function
# -------------------------
@app.function(
    image=image,
    gpu="A10G",   # can also use "L4", "T4", "A100"
    timeout=60 * 60 * 6,
)
def run_gptq(
    model: str = "facebook/opt-125m",
    dataset: str = "wikitext2",
    wbits: int = 4,
    groupsize: int = 128,
):

    import subprocess
    import os

    repo_path = "/workspace/gptq"

    # -------------------------
    # IMPORTANT: run inside repo
    # -------------------------
    cmd = [
        "python",
        "opt.py",   # or opt_new.py if you renamed it
        model,
        dataset,
        "--wbits", str(wbits),
        "--groupsize", str(groupsize),
        "--save", "quantized.pt",
    ]

    print("\n==============================")
    print("Running GPTQ with config:")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Bits: {wbits}")
    print(f"Groupsize: {groupsize}")
    print("==============================\n")

    result = subprocess.run(
        cmd,
        cwd=repo_path,      # 🔥 CRITICAL FIX
        capture_output=True,
        text=True
    )

    print("===== STDOUT =====")
    print(result.stdout)

    print("===== STDERR =====")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("GPTQ failed (see logs above)")

    return "Done"