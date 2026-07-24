#!/usr/bin/env python3
"""
SOTA Real-Experiment v2 (NumPy-only, bug-fixed).

Fixes from v1:
  - SR: proper stochastic rounding in quantization domain (not additive noise)
  - KLD: use smooth-histogram with same bins + KL(P_orig || P_quant)
  - Output KLD: compare W@X distributions properly
  - Add 8-bit reference for calibration
  - Real GPT-2 weights via safetensors (if downloadable)
"""
import os, sys, json, time, math
from collections import OrderedDict
import numpy as np

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ============================================================
# Real weight loading (best-effort)
# ============================================================
def load_gpt2():
    """Try real GPT-2 weights; fallback to realistic synthetic."""
    # Try torch path first
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("[INFO] Loading GPT-2 via torch...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        weights = {}
        for n, p in model.named_parameters():
            weights[n] = p.data.detach().cpu().numpy().astype(np.float32)
        config = dict(model.config.__dict__)
        return weights, config, tok, "GPT-2 (124M, real)"
    except Exception as e:
        print(f"[INFO] torch path failed: {e}")

    # Try safetensors path
    try:
        from huggingface_hub import hf_hub_download
        import struct as st, json as j
        repo = "openai-community/gpt2"
        cfg_path = hf_hub_download(repo_id=repo, filename="config.json")
        with open(cfg_path) as f: config = j.load(f)
        st_path = hf_hub_download(repo_id=repo, filename="model.safetensors")
        with open(st_path, "rb") as f:
            hs = st.unpack("<Q", f.read(8))[0]
            hdr = j.loads(f.read(hs).decode("utf-8"))
            ds = 8 + hs
            weights = {}
            for key, val in hdr.items():
                if key == "__metadata__" or not isinstance(val, dict): continue
                if "shape" not in val or "data_type" not in val: continue
                shape = val["shape"]
                dt_str = val["data_type"]
                off = val.get("data_offsets", [0,0])
                f.seek(ds + off[0])
                raw = f.read(off[1] - off[0])
                dt = {"F32": np.float32, "F16": np.float16, "I32": np.int32}.get(dt_str, np.float32)
                weights[key] = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)
        try:
            from transformers import GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained("gpt2")
        except: tok = None
        return weights, config, tok, "GPT-2 (124M, real, numpy)"
    except Exception as e:
        print(f"[INFO] safetensors path failed: {e}")

    # Fallback: realistic GPT-2 shapes
    print("[INFO] Using realistic GPT-2-shaped synthetic weights")
    config = {"n_layer": 12, "n_embd": 768, "vocab_size": 50257, "n_head": 12}
    weights = {}
    def g(*sh, s=0.02): return rng.standard_normal(sh).astype(np.float32) * s
    weights["wte.weight"] = g(50257, 768)
    weights["transformer/ln_f.weight"] = g(768)
    for L in range(12):
        weights[f"transformer/h.{L}.ln_1.weight"] = g(768)
        weights[f"transformer/h.{L}.ln_2.weight"] = g(768)
        weights[f"transformer/h.{L}.attn.c_attn.weight"] = g(2304, 768)
        weights[f"transformer/h.{L}.attn.c_proj.weight"] = g(768, 768)
        weights[f"transformer/h.{L}.mlp.c_fc.weight"] = g(3072, 768)
        weights[f"transformer/h.{L}.mlp.c_proj.weight"] = g(768, 3072)
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
    except: tok = None
    return weights, config, tok, "GPT-2-shaped (synthetic, 125M)"


# ============================================================
# Quantization (NumPy, correct SR)
# ============================================================

def rtn_quantize(W, bits=4):
    """Symmetric RTN. Returns W_hat, info."""
    W = W.astype(np.float32)
    mx = np.max(np.abs(W))
    if mx < 1e-12: return W.copy(), {"method":"rtn","bits":bits}
    levels = 2**bits - 1
    scale = levels / mx
    q = np.clip(np.round(W * scale), -levels, levels).astype(np.int32)
    return (q.astype(np.float32) / scale), {"method":"rtn","bits":bits,"mx":float(mx)}


def gptq_sim_quantize(W, bits=4):
    """Per-column scaling + RTN (GPTQ essence without Hessian)."""
    W = W.astype(np.float32)
    n_out, n_in = W.shape
    levels = 2**bits - 1
    for j in range(n_in):
        col = W[:, j]
        mx = np.max(np.abs(col))
        sc = levels / max(mx, 1e-12)
        W[:, j] = np.round(col * sc).astype(np.float32) / sc
    return W, {"method":"gptq_sim","bits":bits}


def awq_sim_quantize(W, bits=4):
    """AWQ: protect high-RMS channels via smooth scaling."""
    W = W.astype(np.float32)
    rms = np.sqrt(np.mean(W**2, axis=0) + 1e-12)
    s = (rms / max(np.mean(rms), 1e-12)).clip(0.1, 10.0)
    W_s = W / s[None, :]
    mx = np.max(np.abs(W_s))
    levels = 2**bits - 1
    sc = levels / max(mx, 1e-12)
    q = np.clip(np.round(W_s * sc), -(2**(bits-1)), 2**(bits-1)-1)
    return (q.astype(np.float32) / sc * s[None, :]), {"method":"awq_sim","bits":bits}


def _kurtosis_alpha(W):
    """Per-row alpha from kurtosis (OPQ core)."""
    n = W.shape[0]
    alphas = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = W[i]
        m, sd = np.mean(row), np.std(row) + 1e-8
        kurt = float(np.mean(((row - m) / sd) ** 4)) - 3.0
        alphas[i] = float(np.clip(0.45 + 0.05 * np.tanh(kurt / 10.0), 0.35, 0.65))
    return alphas


def opq_quantize(W, bits=4, per_channel=True, stochastic=False, alpha_override=None):
    """
    Correct OPQ:
    - Quantization: sign * round( (abs(W)/mx)^alpha * levels )
    - Stochastic rounding: in quantized integer domain, add Bernoulli noise to fractional part
    - Dequantization: sign * (q/levels)^(1/alpha) * mx
    """
    W = W.astype(np.float32)
    n_out, n_in = W.shape
    levels = 2**bits - 1

    if per_channel:
        alphas = np.full(n_out, float(alpha_override), dtype=np.float32) if alpha_override else _kurtosis_alpha(W)
        mx = np.max(np.abs(W), axis=1).clip(1e-8)
        norm = np.abs(W) / mx[:, None]
        norm = np.clip(norm, 1e-12, 1.0)
        a_exp = alphas[:, None]
        powered = norm ** a_exp  # [0,1]
        if stochastic:
            # Proper SR: prob = fractional part, add 1 with that probability
            frac = powered * levels - np.floor(powered * levels)
            rand = rng.random(powered.shape)
            add = (rand < frac).astype(np.float32)
            q_f = np.floor(powered * levels) + add
        else:
            q_f = np.floor(powered * levels + 0.5)
        q = np.clip(q_f, 0, levels).astype(np.int32)
        # Dequant
        qn = np.clip(q.astype(np.float32) / levels, 1e-12, 1.0)
        inv = qn ** (1.0 / a_exp)
        W_hat = inv * mx[:, None]
        neg = W < 0
        W_hat[neg] = -np.abs(W_hat[neg])
        tag = "opq_pc_sr" if stochastic else "opq_pc"
        return W_hat, {"method": tag, "bits": bits, "alphas": alphas, "mx": mx}
    else:
        alpha = float(alpha_override) if alpha_override else 0.45
        mx = float(np.max(np.abs(W)))
        norm = np.clip(np.abs(W) / max(mx, 1e-12), 1e-12, 1.0)
        powered = norm ** alpha
        if stochastic:
            frac = powered * levels - np.floor(powered * levels)
            rand = rng.random(powered.shape)
            add = (rand < frac).astype(np.float32)
            q_f = np.floor(powered * levels) + add
        else:
            q_f = np.floor(powered * levels + 0.5)
        q = np.clip(q_f, 0, levels).astype(np.int32)
        qn = np.clip(q.astype(np.float32) / levels, 1e-12, 1.0)
        inv = qn ** (1.0 / alpha)
        W_hat = inv * mx
        neg = W < 0
        W_hat[neg] = -np.abs(W_hat[neg])
        tag = "opq_sr" if stochastic else "opq_global"
        return W_hat, {"method": tag, "bits": bits, "alpha": alpha, "mx": mx}


# ============================================================
# Metrics
# ============================================================

def retention(W, Wh):
    num = float(np.sum((W - Wh)**2))
    den = float(np.sum(W**2)) + 1e-12
    return 1.0 - num / den

def kld_smooth(a, b, bins=512):
    """Smoothed KLD(P_a || P_b) via shared-bin histogram."""
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi - lo < 1e-12: return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    eps = 1e-10
    ha = ha + eps; hb = hb + eps
    ha /= ha.sum(); hb /= hb.sum()
    # KLD(P_orig || P_quant): penalize quant for putting mass where orig doesn't
    return float(np.sum(ha * np.log(ha / hb)))

def output_kld(W, Wh, n=256, in_dim=None):
    """KLD of W@X vs Wh@X for random X, averaged over output rows."""
    in_dim = in_dim or W.shape[1]
    X = rng.standard_normal((n, in_dim)).astype(np.float32) * 0.02
    y_o = W.astype(np.float32) @ X.T
    y_h = Wh.astype(np.float32) @ X.T
    klds = []
    for i in range(min(y_o.shape[0], 128)):
        k = kld_smooth(y_o[i], y_h[i], bins=128)
        klds.append(k)
    return float(np.mean(klds))

def compression(bits, n_params):
    return 16.0 / bits


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    weights, config, tok, model_name = load_gpt2()
    total_params = sum(w.size for w in weights.values())
    print(f"[OK] {model_name}, {total_params/1e6:.1f}M params, {len(weights)} tensors")

    # Pick linear layers (2D, >1K params)
    layers = [(n, W) for n, W in weights.items() if W.ndim == 2 and W.size > 1000]
    print(f"[OK] Evaluating {len(layers)} linear layers")

    methods = OrderedDict([
        ("FP16",          {"bits":16, "fn":None}),
        ("RTN-8bit",      {"bits":8,  "fn":lambda W: rtn_quantize(W,8)}),
        ("RTN-4bit",      {"bits":4,  "fn":lambda W: rtn_quantize(W,4)}),
        ("GPTQsim-4bit",  {"bits":4,  "fn":lambda W: gptq_sim_quantize(W,4)}),
        ("AWQsim-4bit",   {"bits":4,  "fn":lambda W: awq_sim_quantize(W,4)}),
        ("OPQ-Global",    {"bits":4,  "fn":lambda W: opq_quantize(W,4,per_channel=False,stochastic=False,alpha_override=0.45)}),
        ("OPQ-PC",        {"bits":4,  "fn":lambda W: opq_quantize(W,4,per_channel=True,stochastic=False)}),
        ("OPQ-PC-SR",     {"bits":4,  "fn":lambda W: opq_quantize(W,4,per_channel=True,stochastic=True)}),
        ("OPQ-MoE-4b",    {"bits":4,  "fn":lambda W: opq_quantize(W,4,per_channel=True,stochastic=True,alpha_override=0.55)}),
        ("OPQ-MoE-3b",    {"bits":3,  "fn":lambda W: opq_quantize(W,3,per_channel=True,stochastic=True,alpha_override=0.60)}),
    ])

    # Run
    all_results = []
    for mname, mcfg in methods.items():
        print(f"\n{'='*65}\n  {mname}\n{'='*65}")
        layer_recs = []
        for lname, W in layers:
            if mcfg["fn"] is None:
                Wh, info = W.copy(), {"method":"fp16"}
            else:
                Wh, info = mcfg["fn"](W)
            ret = retention(W, Wh)
            rel = float(np.sqrt(np.mean((W-Wh)**2)) / (np.std(W)+1e-8))
            kldw = kld_smooth(W.flatten(), Wh.flatten(), bins=512)
            kldo = output_kld(W, Wh, in_dim=W.shape[1])
            rec = {"layer":lname,"params":W.size,"ret":ret,"rel":rel,"kldw":kldw,"kldo":kldo}
            layer_recs.append(rec)
            if len(layer_recs) <= 4 or "lm_head" in lname or lname == "wte.weight":
                print(f"  {lname:50s} ret={ret*100:6.3f}%  rel={rel:6.4f}  KLDw={kldw:8.5f}  KLDout={kldo:8.5f}")

        n = len(layer_recs)
        wp = sum(r["params"] for r in layer_recs)
        w_ret = sum(r["ret"]*r["params"] for r in layer_recs)/max(wp,1)
        avg_ret = float(np.mean([r["ret"] for r in layer_recs]))
        avg_rel = float(np.mean([r["rel"] for r in layer_recs]))
        avg_kldw = float(np.mean([r["kldw"] for r in layer_recs]))
        avg_kldo = float(np.mean([r["kldo"] for r in layer_recs if r["kldo"]>0]))
        comp = compression(mcfg["bits"], wp)
        summary = {
            "method":mname,"bits":mcfg["bits"],"n_layers":n,
            "total_params":wp,"weighted_retention":w_ret,
            "avg_retention":avg_ret,"avg_rel_err":avg_rel,
            "avg_kld_weights":avg_kldw,"avg_kld_output":avg_kldo,
            "compression":comp,"elapsed_s":time.time()-t0,
            "layer_samples":layer_recs[:8]
        }
        all_results.append(summary)
        print(f"  >> {mname}: WtdRet={w_ret*100:.3f}%, AvgKLDw={avg_kldw:.5f}, AvgKLDout={avg_kldo:.5f}, Compress={comp:.2f}x")

    elapsed = time.time() - t0

    # ---- Summary ----
    print(f"\n{'='*100}")
    print(f"  SOTA COMPARISON — {model_name}  ({total_params/1e6:.1f}M params, {elapsed:.0f}s)")
    print(f"{'='*100}")
    print(f"{'Method':<18s} {'B':>3s} {'WtdRet%':>10s} {'AvgRet%':>10s} {'RelErr':>8s} {'KLDw':>10s} {'KLDout':>10s} {'CompX':>7s}")
    print("-"*100)
    for r in all_results:
        print(f"{r['method']:<18s} {r['bits']:>3d} {r['weighted_retention']*100:>9.3f}% {r['avg_retention']*100:>9.3f}% "
              f"{r['avg_rel_err']:>8.4f} {r['avg_kld_weights']:>10.5f} {r['avg_kld_output']:>10.5f} {r['compression']:>6.2f}x")

    # ---- Save JSON ----
    out = "/data/workspace/sota_results"
    os.makedirs(out, exist_ok=True)
    payload = {"model":model_name,"config":str(config),"total_params":total_params,
               "elapsed_s":elapsed,"results":all_results}
    with open(f"{out}/sota_results.json","w") as f: json.dump(payload,f,indent=2,default=str)

    # ---- Markdown ----
    md = f"## SOTA Quantization — {model_name}\n\n**Params**: {total_params/1e6:.1f}M | **Time**: {elapsed:.0f}s\n\n"
    md += "| Method | B | WtdRet% | AvgRet% | RelErr | KLDw | KLDout | CompX |\n|---|---|---|---|---|---|---|---|\n"
    for r in all_results:
        md += f"| {r['method']} | {r['bits']} | {r['weighted_retention']*100:.3f} | {r['avg_retention']*100:.3f} | {r['avg_rel_err']:.4f} | {r['avg_kld_weights']:.5f} | {r['avg_kld_output']:.5f} | {r['compression']:.2f} |\n"
    with open(f"{out}/sota_results.md","w") as f: f.write(md)

    # ---- LaTeX table ----
    latex = "% Auto-generated SOTA table\n\\begin{table}[h]\n\\centering\n"
    latex += f"\\caption{{SOTA comparison on {model_name} ({total_params/1e6:.0f}M params).}}\\n"
    latex += "\\label{tab:sota}\\begin{tabular}{lcccccc}\n\\toprule\n"
    latex += "Method & Bits & WtdRet\\% & RelErr & KLD$_w$ & KLD$_{out}$ & CompX \\\\\n\\midrule\n"
    for r in all_results:
        latex += f"{r['method']} & {r['bits']} & {r['weighted_retention']*100:.2f} & {r['avg_rel_err']:.4f} & {r['avg_kld_weights']:.5f} & {r['avg_kld_output']:.5f} & {r['compression']:.2f} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(f"{out}/sota_table.tex","w") as f: f.write(latex)

    # ---- Generation samples (if tokenizer) ----
    if tok is not None:
        prompts = ["The future of AI",
                   "In a groundbreaking",
                   "The capital of France"]
        samples = {}
        # Use lm_head or wte
        W_lm = weights.get("lm_head.weight") or weights.get("wte.weight")
        if W_lm is not None:
            for mname, mcfg in methods.items():
                if mcfg["fn"] is None: continue
                try:
                    Wh, _ = mcfg["fn"](W_lm)
                    outs = []
                    for p in prompts:
                        ids = tok.encode(p, return_tensors="np")[0]
                        last = ids[-1]
                        scores = Wh @ np.eye(Wh.shape[1], dtype=np.float32)[last]
                        top5 = np.argsort(scores)[-5:][::-1]
                        toks = [tok.decode([t]) for t in top5]
                        outs.append(f"{p} -> {toks}")
                    samples[mname] = outs
                except Exception as e:
                    samples[mname] = [f"ERROR: {e}"]
            with open(f"{out}/generation_samples.json","w") as f: json.dump(samples,f,indent=2)

    print(f"\n[OK] Saved to {out}/")


if __name__ == "__main__":
    main()
