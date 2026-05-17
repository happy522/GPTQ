import argparse
import math
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptq import GPTQ
from modelutils import find_layers
from quant import Quant3Linear, Quantizer, make_quant3, quantize
from datautils import get_loaders

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Family detection / model IO
# -----------------------------
def detect_family(model_name: str, config=None) -> str:
    name = model_name.lower()
    model_type = getattr(config, "model_type", "").lower() if config is not None else ""
    archs = [a.lower() for a in (getattr(config, "architectures", None) or [])]

    if "opt" in name or model_type == "opt" or any("opt" in a for a in archs):
        return "opt"
    if "qwen" in name or "qwen" in model_type or any("qwen" in a for a in archs):
        return "qwen"
    if "llama" in name or "llama" in model_type or any("llama" in a for a in archs):
        return "llama"

    raise ValueError(
        f"Could not infer model family from '{model_name}'. "
        "Use an OPT, Llama, or Qwen checkpoint."
    )


def load_model(model_name: str, trust_remote_code: bool = False):
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    family = detect_family(model_name, config)

    def skip_init(*args, **kwargs):
        pass

    # Match the original GPTQ repo behavior.
    torch.nn.init.kaiming_uniform_ = skip_init
    torch.nn.init.uniform_ = skip_init
    torch.nn.init.normal_ = skip_init

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.seqlen = getattr(model.config, "max_position_embeddings", 2048)
    return model, family


def get_stack(model, family: str):
    if family == "opt":
        stack = model.model.decoder
        return SimpleNamespace(
            stack=stack,
            layers=stack.layers,
            embed_tokens=stack.embed_tokens,
            embed_positions=getattr(stack, "embed_positions", None),
            final_norm=getattr(stack, "final_layer_norm", None),
            project_in=getattr(stack, "project_in", None),
            project_out=getattr(stack, "project_out", None),
            layer_prefix="model.decoder.layers",
        )

    # Llama / Qwen-family decoder-only stacks
    stack = model.model
    return SimpleNamespace(
        stack=stack,
        layers=stack.layers,
        embed_tokens=stack.embed_tokens,
        embed_positions=None,
        final_norm=getattr(stack, "norm", None),
        project_in=None,
        project_out=None,
        layer_prefix="model.layers",
    )


def maybe_to(module, dev):
    if module is not None:
        return module.to(dev)
    return None


def run_layer(module, x, attention_mask=None):
    if attention_mask is None:
        return module(x)[0]
    return module(x, attention_mask=attention_mask)[0]


# -----------------------------
# Quantization
# -----------------------------
@torch.no_grad()
def sequential_gptq(model, dataloader, dev, family, args):
    print("Starting ...")

    parts = get_stack(model, family)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    parts.embed_tokens = maybe_to(parts.embed_tokens, dev)
    parts.embed_positions = maybe_to(parts.embed_positions, dev)
    parts.project_in = maybe_to(parts.project_in, dev)
    parts.project_out = maybe_to(parts.project_out, dev)

    layers = parts.layers
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    parts.embed_tokens = parts.embed_tokens.cpu()
    if parts.embed_positions is not None:
        parts.embed_positions = parts.embed_positions.cpu()
    if parts.project_in is not None:
        parts.project_in = parts.project_in.cpu()
    if parts.project_out is not None:
        parts.project_out = parts.project_out.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}

        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits,
                perchannel=True,
                sym=args.sym,
                mse=False,
                trits=args.trits,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]

        for j in range(args.nsamples):
            outs[j] = run_layer(layer, inps[j].unsqueeze(0), attention_mask)

        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print("Quantizing ...")
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.groupsize,
                actorder=args.act_order,
                static_groups=args.static_groups,
            )
            quantizers[f"{parts.layer_prefix}.{i}.{name}"] = gptq[name].quantizer
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = run_layer(layer, inps[j].unsqueeze(0), attention_mask)

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_ppl(model, testenc, dev, family):
    print("Evaluating ...")

    parts = get_stack(model, family)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    parts.embed_tokens = maybe_to(parts.embed_tokens, dev)
    parts.embed_positions = maybe_to(parts.embed_positions, dev)
    parts.project_in = maybe_to(parts.project_in, dev)
    parts.project_out = maybe_to(parts.project_out, dev)

    layers = parts.layers
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if parts.embed_tokens is not None:
        parts.embed_tokens = parts.embed_tokens.cpu()
    if parts.embed_positions is not None:
        parts.embed_positions = parts.embed_positions.cpu()
    if parts.project_in is not None:
        parts.project_in = parts.project_in.cpu()
    if parts.project_out is not None:
        parts.project_out = parts.project_out.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = run_layer(layer, inps[j].unsqueeze(0), attention_mask)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if parts.final_norm is not None:
        parts.final_norm = parts.final_norm.to(dev)
    if parts.project_out is not None:
        parts.project_out = parts.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []

    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if parts.final_norm is not None:
            hidden_states = parts.final_norm(hidden_states)
        if parts.project_out is not None:
            hidden_states = parts.project_out(hidden_states)

        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    ppl_value = ppl.item()
    print(f"Perplexity: {ppl_value:.4f}")

    model.config.use_cache = use_cache
    return ppl_value


# -----------------------------
# Save packed quantized model
# -----------------------------
def save_quantized(model, quantizers, args):
    print("Packing ...")
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])

    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(
            qlayers[name].weight if hasattr(qlayers[name], "weight") else find_layers(model)[name],
            quantizers[name].scale,
            quantizers[name].zero,
        )

    print("Done.")
    torch.save(model.state_dict(), args.save)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Any OPT, Llama, or Qwen-family causal LM checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2"],
        help="Calibration/eval data.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--nearest", action="store_true")
    parser.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 16])
    parser.add_argument("--trits", action="store_true")
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--faster-kernel", action="store_true")
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--static-groups", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Use this only if a checkpoint needs custom HF code.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GPTQ CONFIG")
    print("=" * 60)
    print(f"Model     : {args.model}")
    print(f"Dataset   : {args.dataset}")
    print(f"Bits      : {args.wbits}")
    print(f"Groupsize : {args.groupsize}")
    print(f"Samples   : {args.nsamples}")
    print(f"Device    : {DEV}")
    print("=" * 60)

    if args.load:
        raise NotImplementedError("Keep loading separate for now; quantize first, then load.")
    else:
        model, family = load_model(args.model, trust_remote_code=args.trust_remote_code)

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    quantizers = None
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = sequential_gptq(model, dataloader, DEV, family, args)
        print(f"Quantization completed in {time.time() - tick:.2f}s")

    results = {}
    ppl = evaluate_ppl(model, testloader, DEV, family)
    results[args.dataset] = ppl

    if args.save:
        if quantizers is None:
            raise RuntimeError("Nothing to save because quantization did not run.")
        save_quantized(model, quantizers, args)

    print("\nRESULTS:")
    for ds, val in results.items():
        print(f"  {ds}: Perplexity = {val:.4f}")


if __name__ == "__main__":
    main()