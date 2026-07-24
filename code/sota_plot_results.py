#!/usr/bin/env python3
"""Plot SOTA comparison charts from sota_results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "WenQuanYi Micro Hei"
rcParams["axes.unicode_minus"] = False

IN = "/data/workspace/sota_results/sota_results.json"
OUT = "/data/workspace/sota_results"
os.makedirs(OUT, exist_ok=True)

with open(IN) as f:
    data = json.load(f)

model = data["model"]
fp16_ppl = data["fp16_ppl"]
results = data["results"]

# Filter successful
ok = [r for r in results if r.get("ppl", float("inf")) < float("inf")]

labels = [r["label"] for r in ok]
ppls = [r["ppl"] for r in ok]
diffs = [r["ppl_diff"] for r in ok]
rets = [r["retention_pct"] for r in ok]
errs = [r["avg_rel_err"] for r in ok]
klds = [r.get("kld", 0) for r in ok]
bits = [r["bits"] for r in ok]

colors = []
for r in ok:
    m = r["method"]
    if "fp16" in m: colors.append("#2ecc71")
    elif "opq_moe" in m: colors.append("#e74c3c")
    elif "opq_pc_sr" in m: colors.append("#e67e22")
    elif "opq_pc" in m: colors.append("#f39c12")
    elif "opq" in m: colors.append("#3498db")
    elif "gptq" in m: colors.append("#9b59b6")
    elif "awq" in m: colors.append("#1abc9c")
    else: colors.append("#95a5a6")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f"SOTA Quantization Comparison on {model}\nFP16 Baseline PPL = {fp16_ppl:.4f}",
             fontsize=16, fontweight="bold")

# (a) PPL bar
ax = axes[0, 0]
bars = ax.barh(range(len(labels)), ppls, color=colors, edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(fp16_ppl, color="black", linestyle="--", linewidth=1.5, label=f"FP16={fp16_ppl:.2f}")
ax.set_xlabel("Perplexity (WikiText-2, lower is better)")
ax.set_title("(a) Perplexity Comparison")
ax.legend(loc="lower right")
for i, v in enumerate(ppls):
    ax.text(v + 0.05, i, f"{v:.3f}", va="center", fontsize=8)

# (b) PPL diff
ax = axes[0, 1]
bars = ax.barh(range(len(labels)), diffs, color=colors, edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color="black", linestyle="-", linewidth=1)
ax.set_xlabel("ΔPPL (lower is better)")
ax.set_title("(b) Perplexity Degradation")

# (c) Retention
ax = axes[0, 2]
bars = ax.barh(range(len(labels)), rets, color=colors, edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(99.0, color="red", linestyle="--", linewidth=1.5, label="99% target")
ax.axvline(99.9, color="darkred", linestyle=":", linewidth=1.5, label="99.9% target")
ax.set_xlim(90, 100.5)
ax.set_xlabel("Weight Retention % (higher is better)")
ax.set_title("(c) Performance Retention")
ax.legend(loc="lower left")

# (d) Avg rel error
ax = axes[1, 0]
ax.barh(range(len(labels)), [e * 100 for e in errs], color=colors, edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Avg Relative Error (%)")
ax.set_title("(d) Quantization Error")
ax.set_xscale("log")

# (e) KLD
ax = axes[1, 1]
kld_vals = [k if k == k else 0 for k in klds]  # nan->0
bars = ax.barh(range(len(labels)), kld_vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(0.6, color="gray", linestyle="--", linewidth=1.5, label="KLD=0.6 ref")
ax.set_xlabel("KLD (lower is better)")
ax.set_title("(e) Output Distribution KLD")
ax.legend(loc="lower right")

# (f) Bits vs Retention scatter (Pareto)
ax = axes[1, 2]
for i, r in enumerate(ok):
    ax.scatter(bits[i], rets[i], s=200, color=colors[i], edgecolor="black", linewidth=1.5, zorder=5)
    ax.annotate(r["label"], (bits[i], rets[i]), fontsize=7, ha="left",
                xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("Quantization Bits")
ax.set_ylabel("Retention %")
ax.set_title("(f) Bits vs Retention (Pareto Frontier)")
ax.set_xlim(2.5, 16.5)
ax.set_ylim(90, 101)
# Mark OPQ-MoE
opq_moe = [r for r in ok if "OPQ-MoE" in r["label"]]
if opq_moe:
    r = opq_moe[0]
    ax.annotate("← Pareto point", (r["bits"], r["retention_pct"]),
                fontsize=9, color="red", fontweight="bold",
                xytext=(-60, 10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"))

plt.tight_layout()
out_png = os.path.join(OUT, "sota_comparison.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"[OK] Saved {out_png}")

# Also save per-method JSON for LaTeX table
tbl = {"model": model, "fp16_ppl": fp16_ppl, "methods": ok}
with open(os.path.join(OUT, "sota_table.json"), "w") as f:
    json.dump(tbl, f, indent=2)
print(f"[OK] Saved {os.path.join(OUT, 'sota_table.json')}")
