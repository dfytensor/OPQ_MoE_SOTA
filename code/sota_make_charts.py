#!/usr/bin/env python3
"""Generate 6-panel SOTA comparison chart from sota_results.json."""
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
with open(IN) as f: data = json.load(f)

model = data["model"]
results = data["results"]

labels = [r["method"] for r in results]
wret = [r["weighted_retention"]*100 for r in results]
aret = [r["avg_retention"]*100 for r in results]
rel = [r["avg_rel_err"] for r in results]
kldw = [r["avg_kld_weights"] for r in results]
kldo = [r["avg_kld_output"] for r in results]
bits = [r["bits"] for r in results]
comp = [r["compression"] for r in results]

# Colors
def color_for(m):
    if "FP16" in m: return "#27ae60"
    if "OPQ-PC-SR" in m: return "#e74c3c"
    if "OPQ-PC" in m and "SR" not in m: return "#e67e22"
    if "OPQ-MoE" in m: return "#c0392b"
    if "OPQ" in m: return "#f39c12"
    if "GPTQ" in m: return "#8e44ad"
    if "AWQ" in m: return "#16a085"
    if "RTN-8" in m: return "#3498db"
    return "#7f8c8d"
colors = [color_for(m) for m in labels]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f"SOTA Quantization Comparison on {model}\n"
             f"({data['total_params']/1e6:.0f}M params, {data['elapsed_s']:.0f}s)",
             fontsize=15, fontweight="bold")

# (a) Weighted Retention bar
ax = axes[0, 0]
y = np.arange(len(labels))
ax.barh(y, wret, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(99.0, color="red", linestyle="--", lw=1.5, label="99% target")
ax.axvline(99.9, color="darkred", linestyle=":", lw=1.5, label="99.9% target")
ax.set_xlabel("Weighted Retention % (higher better)")
ax.set_title("(a) Weighted Retention")
ax.set_xlim(90, 101)
ax.legend(loc="lower left", fontsize=8)
for i, v in enumerate(wret):
    ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)

# (b) KLD weights (log)
ax = axes[0, 1]
ax.barh(y, kldw, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("KLD$_w$ (lower better)")
ax.set_title("(b) Weight Distribution KLD")
ax.set_xscale("log")
ax.axvline(1.0, color="gray", linestyle="--", lw=1, label="KLD=1")
ax.legend(fontsize=8)
for i, v in enumerate(kldw):
    ax.text(v * 1.1, i, f"{v:.3f}", va="center", fontsize=8)

# (c) Output KLD
ax = axes[0, 2]
ax.barh(y, kldo, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("KLD$_{out}$ (lower better)")
ax.set_title("(c) Output Distribution KLD (W@X)")
ax.axvline(2.0, color="gray", linestyle="--", lw=1, label="KLD=2 ref")
ax.legend(fontsize=8)
for i, v in enumerate(kldo):
    ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)

# (d) Relative Error
ax = axes[1, 0]
ax.barh(y, rel, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Normalized Relative Error (lower better)")
ax.set_title("(d) Reconstruction Error")
ax.set_xscale("log")

# (e) Compression vs Retention scatter (Pareto)
ax = axes[1, 1]
for i in range(len(labels)):
    ax.scatter(comp[i], wret[i], s=200, color=colors[i], edgecolor="black", lw=1.5, zorder=5)
    ax.annotate(labels[i], (comp[i], wret[i]), fontsize=7, ha="left",
                xytext=(5, 3), textcoords="offset points")
ax.set_xlabel("Compression Ratio (vs FP16)")
ax.set_ylabel("Weighted Retention %")
ax.set_title("(e) Pareto: Compression vs Retention")
ax.set_xlim(0.5, 6)
ax.set_ylim(93, 101)
# Mark best
best = max(zip(wret, range(len(wret))))[1]
ax.annotate("← Best 4-bit", (comp[best], wret[best]), fontsize=9, color="red",
            fontweight="bold", xytext=(-70, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="red"))

# (f) Bit-width grouped retention
ax = axes[1, 2]
# Group by bits
from collections import defaultdict
groups = defaultdict(list)
for i, b in enumerate(bits):
    groups[b].append(wret[i])
for i in range(len(labels)):
    ax.scatter(bits[i] + (i%3-1)*0.08, wret[i], s=120, color=colors[i], edgecolor="black", lw=1, zorder=5)
ax.set_xticks(sorted(groups.keys()))
ax.set_xlabel("Quantization Bits")
ax.set_ylabel("WtdRet %")
ax.set_title("(f) Bits vs Retention")
ax.set_ylim(93, 101)

plt.tight_layout()
out_png = os.path.join(OUT, "sota_comparison.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"[OK] {out_png}")

# Also save figures/ copy
shutil_dest = "/data/workspace/sota_results"
import shutil
shutil.copy2(out_png, os.path.join(shutil_dest, "sota_comparison.png"))
print("[OK] copied to sota_results/")
