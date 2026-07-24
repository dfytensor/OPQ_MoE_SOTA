"""
ROQ v3: Dual-Tower + Per-Channel Quantization (Core Experiment)
================================================================
From "10000=100^2" to +71.6% MSE reduction on 4-bit weight quantization.

This script reproduces all key results of the ROQ v3 framework:
  - Method comparison (6 methods, attn_q layer, 4-bit)
  - Per-layer breakdown (4 Transformer layers)
  - End-to-end Transformer block simulation
  - Large-scale validation (1024x1024)
  - Master figure with 6 subplots

Run:  python ropq_v3_run.py
Time: ~3 minutes on CPU
Deps: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================
# OPQ Quantizer
# ============================================================
class OPQ:
    """Optimal Power Quantization: q = Round(|w|^alpha * s)"""
    def __init__(self, bits=4, alpha=0.45):
        self.bits = bits
        self.alpha = alpha
        self.L = 1 << (bits - 1)   # magnitude levels (e.g. 8 for 4-bit)
        self.s = None               # scale factor

    def cal(self, W):
        """Calibrate scale from weight tensor"""
        mx = np.max(np.abs(W))
        if mx == 0: mx = 1.0
        self.s = (self.L - 1) / (mx ** self.alpha)
        return self

    def q(self, W):
        """Quantize: w -> integer"""
        z = np.abs(W) ** self.alpha * self.s
        qm = np.clip(np.round(z), 0, self.L - 1).astype(np.int64)
        # Pack sign in MSB
        return ((qm) | ((W < 0).astype(np.int64) << (self.bits - 1))).astype(np.uint8)

    def dq(self, q):
        """Dequantize: integer -> w_hat"""
        sb = 1 << (self.bits - 1)
        sign = np.where((q & sb) != 0, -1.0, 1.0)
        qm = (q & (sb - 1)).astype(np.float64)
        safe = np.clip(qm / self.s, 1e-30, None)
        return sign * (safe ** (1.0 / self.alpha))


def best_a(ch, bits):
    """Search optimal alpha for a single channel"""
    best_m = float('inf')
    best_alpha = None
    for a in [0.30, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45, 0.48, 0.50, 0.53, 0.55, 0.58]:
        q = OPQ(bits, a)
        q.cal(ch)
        m = np.mean((ch - q.dq(q.q(ch))) ** 2)
        if m < best_m:
            best_m = m
            best_alpha = a
    return best_alpha


def perch(W, bits):
    """Per-channel alpha search quantization"""
    Wh = np.zeros_like(W)
    for i in range(W.shape[0]):
        a = best_a(W[i], bits)
        q = OPQ(bits, a)
        q.cal(W[i])
        Wh[i] = q.dq(q.q(W[i]))
    return Wh


def dt_perch(W, bits):
    """
    Dual-Tower + Per-Channel: the ultimate ROQ v3 method.
    Split weights by magnitude threshold tau, then per-channel search
    optimal alpha in each tower independently.
    """
    # Search optimal tau
    best_tau = None
    best_m = float('inf')
    for tau in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050]:
        Wh = np.zeros_like(W)
        lm = np.abs(W) >= tau
        sm = np.abs(W) < tau
        if lm.sum() > 0:
            for i in range(W.shape[0]):
                m = lm[i]
                if m.sum() > 0:
                    ch = W[i, m]
                    a = best_a(ch, bits)
                    q = OPQ(bits, a)
                    q.cal(ch)
                    Wh[i, m] = q.dq(q.q(ch))
        if sm.sum() > 0:
            for i in range(W.shape[0]):
                m = sm[i]
                if m.sum() > 0:
                    ch = W[i, m]
                    a = best_a(ch, bits)
                    q = OPQ(bits, a)
                    q.cal(ch)
                    Wh[i, m] = q.dq(q.q(ch))
        m = np.mean((W - Wh) ** 2)
        if m < best_m:
            best_m = m
            best_tau = tau

    # Final quantization with best tau
    Wh = np.zeros_like(W)
    lm = np.abs(W) >= best_tau
    sm = np.abs(W) < best_tau
    if lm.sum() > 0:
        for i in range(W.shape[0]):
            m = lm[i]
            if m.sum() > 0:
                ch = W[i, m]
                a = best_a(ch, bits)
                q = OPQ(bits, a)
                q.cal(ch)
                Wh[i, m] = q.dq(q.q(ch))
    if sm.sum() > 0:
        for i in range(W.shape[0]):
            m = sm[i]
            if m.sum() > 0:
                ch = W[i, m]
                a = best_a(ch, bits)
                q = OPQ(bits, a)
                q.cal(ch)
                Wh[i, m] = q.dq(q.q(ch))

    return Wh, best_tau


def genW(n, m, lt, seed):
    """Generate realistic LLM-like weight tensors"""
    np.random.seed(seed)
    if lt == 'q':
        W = np.random.normal(0, 0.02, (n, m))
    elif lt == 'v':
        W = np.random.normal(0, 0.03, (n, m))
    elif lt == 'g':
        W = np.random.laplace(0, 0.04, (n, m))
        W = np.clip(W, -1, 1)
    elif lt == 'u':
        W = np.random.normal(0, 0.04, (n, m))
    else:
        W = np.random.normal(0, 0.03, (n, m))
    np.random.seed(None)
    return W.astype(np.float32)


# ============================================================
# Main Experiment
# ============================================================
print("=" * 70)
print("ROQ v3: Dual-Tower + Per-Channel Quantization")
print("=" * 70)

# --- Build a Transformer block (dim=512, 4 layers) ---
dim = 512
layers = {}
for name, lt, sd in [('attn_q', 'q', 100), ('attn_v', 'v', 101),
                      ('ffn_gate', 'g', 200), ('ffn_up', 'u', 201)]:
    layers[name] = genW(dim, dim, lt, sd)

total = sum(W.size for W in layers.values())
print(f"Total parameters: {total:,}")

# ============================================================
# Exp 1: Method comparison (attn_q, 4-bit)
# ============================================================
print("\n" + "=" * 70)
print("Exp 1: All Methods (attn_q, 4-bit)")
print("=" * 70)

W = layers['attn_q']
sig = np.mean(W ** 2)
base = None
results = {}

# 1. Global alpha=0.45 (baseline)
q = OPQ(4, 0.45)
q.cal(W)
m = np.mean((W - q.dq(q.q(W))) ** 2)
base = m
results['Global OPQ alpha=0.45'] = m
print(f"1. Global alpha=0.45:  MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB")

# 2. Global alpha=0.50
q = OPQ(4, 0.50)
q.cal(W)
m = np.mean((W - q.dq(q.q(W))) ** 2)
results['Global OPQ alpha=0.50'] = m
print(f"2. Global alpha=0.50:  MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB")

# 3. Per-channel alpha search
t0 = time.time()
Wh = perch(W, 4)
m = np.mean((W - Wh) ** 2)
results['Per-channel alpha'] = m
print(f"3. Per-channel:       MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB  "
      f"({(1-m/base)*100:+.1f}%)  [{time.time()-t0:.0f}s]")

# 4. Dual-Tower with global alpha
best_t = None
best_m = float('inf')
for tau in [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.080]:
    lm = np.abs(W) >= tau
    sm = np.abs(W) < tau
    Wh = np.zeros_like(W)
    if lm.sum() > 0:
        q = OPQ(4, 0.50)
        q.cal(W[lm])
        Wh[lm] = q.dq(q.q(W[lm]))
    if sm.sum() > 0:
        q = OPQ(4, 0.25)
        q.cal(W[sm])
        Wh[sm] = q.dq(q.q(Wh[sm]))
    m = np.mean((W - Wh) ** 2)
    if m < best_m:
        best_m = m
        best_t = tau

lm = np.abs(W) >= best_t
sm = np.abs(W) < best_t
Wh = np.zeros_like(W)
if lm.sum() > 0:
    q = OPQ(4, 0.50)
    q.cal(W[lm])
    Wh[lm] = q.dq(q.q(W[lm]))
if sm.sum() > 0:
    q = OPQ(4, 0.25)
    q.cal(W[sm])
    Wh[sm] = q.dq(q.q(W[sm]))
m = np.mean((W - Wh) ** 2)
results[f'Dual-Tower (tau={best_t:.3f})'] = m
print(f"4. Dual-Tower tau={best_t:.3f}: MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB  "
      f"({(1-m/base)*100:+.1f}%)")

# 5. *** DT + Per-channel (the champion) ***
Wh, tau_dp = dt_perch(W, 4)
m = np.mean((W - Wh) ** 2)
results[f'DT+Per-ch (tau={tau_dp:.3f})'] = m
print(f"5. *** DT+Per-ch tau={tau_dp:.3f}: MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB  "
      f"({(1-m/base)*100:+.1f}%) ***")

# 6. 3-bit baseline (limit)
q = OPQ(3, 0.40)
q.cal(W)
m = np.mean((W - q.dq(q.q(W))) ** 2)
results['3-bit alpha=0.40'] = m
print(f"6. 3-bit alpha=0.40:    MSE={m:.2e}  SNR={10*np.log10(sig/(m+1e-30)):.2f}dB")

# ============================================================
# Exp 2: Per-layer breakdown
# ============================================================
print("\n" + "=" * 70)
print("Exp 2: Per-Layer Breakdown (4-bit)")
print("=" * 70)
print(f"{'Layer':>10} | {'Global':>10} | {'Per-ch':>10} | {'DT+PC':>10} | {'Best':>12}")
print("-" * 60)

layer_snrs = {}
for name, Wl in layers.items():
    sigl = np.mean(Wl ** 2)
    # Global
    q = OPQ(4, 0.45)
    q.cal(Wl)
    m1 = np.mean((Wl - q.dq(q.q(Wl))) ** 2)
    # Per-channel
    Wh2 = perch(Wl, 4)
    m2 = np.mean((Wl - Wh2) ** 2)
    # DT+PC
    Wh3, _ = dt_perch(Wl, 4)
    m3 = np.mean((Wl - Wh3) ** 2)

    snrs = [10 * np.log10(sigl / (m + 1e-30)) for m in [m1, m2, m3]]
    layer_snrs[name] = snrs
    best_idx = np.argmax(snrs)
    best_name = ['Global', 'Per-ch', 'DT+PC'][best_idx]
    print(f"  {name:>10} | {m1:.1e} | {m2:.1e} | {m3:.1e} | {best_name} ({snrs[best_idx]:.1f}dB)")

# ============================================================
# Exp 3: End-to-End Transformer Block Simulation
# ============================================================
print("\n" + "=" * 70)
print("Exp 3: End-to-End Inference Simulation")
print("=" * 70)

d = dim
seq = 128
batch = 4
np.random.seed(7)
X = np.random.normal(0, 0.5, (batch, seq, d)).astype(np.float32)

# FP16 reference
Yref = X.copy()
for name, Wl in layers.items():
    Yref = Yref @ Wl.T

print(f"Config: {batch}x{seq}x{d}, {len(layers)} layers")
print(f"\n  {'Method':>25} | {'E2E MSE':>12} | {'SNR':>8}")
print("  " + "-" * 50)

e2e = {}

# Global 4-bit
Ya = X.copy()
for name, Wl in layers.items():
    q = OPQ(4, 0.45)
    q.cal(Wl)
    Ya = Ya @ q.dq(q.q(Wl)).T
msa = np.mean((Yref - Ya) ** 2)
sna = 10 * np.log10(np.mean(Yref ** 2) / (msa + 1e-30))
e2e['Global 4-bit'] = (msa, sna)
print(f"  {'Global 4-bit OPQ':>25} | {msa:>12.2e} | {sna:>7.2f}dB")

# Per-channel
Yb = X.copy()
for name, Wl in layers.items():
    Wh = perch(Wl, 4)
    Yb = Yb @ Wh.T
msb = np.mean((Yref - Yb) ** 2)
snb = 10 * np.log10(np.mean(Yref ** 2) / (msb + 1e-30))
e2e['Per-channel'] = (msb, snb)
print(f"  {'Per-channel alpha':>25} | {msb:>12.2e} | {snb:>7.2f}dB")

# *** DT+Per-ch ***
Yc = X.copy()
for name, Wl in layers.items():
    Wh, _ = dt_perch(Wl, 4)
    Yc = Yc @ Wh.T
msc = np.mean((Yref - Yc) ** 2)
snc = 10 * np.log10(np.mean(Yref ** 2) / (msc + 1e-30))
e2e['DT+Per-ch'] = (msc, snc)
print(f"  {'*** DT+Per-ch ***':>25} | {msc:>12.2e} | {snc:>7.2f}dB")

# 3-bit
Ye = X.copy()
for name, Wl in layers.items():
    q = OPQ(3, 0.40)
    q.cal(Wl)
    Ye = Ye @ q.dq(q.q(Wl)).T
mse3 = np.mean((Yref - Ye) ** 2)
sne = 10 * np.log10(np.mean(Yref ** 2) / (mse3 + 1e-30))
e2e['3-bit'] = (mse3, sne)
print(f"  {'Global 3-bit':>25} | {mse3:>12.2e} | {sne:>7.2f}dB")

# ============================================================
# Exp 4: Large-Scale Validation (1024x1024)
# ============================================================
print("\n" + "=" * 70)
print("Exp 4: Large-Scale Validation (1024x1024)")
print("=" * 70)

for lt, sd in [('q', 300), ('g', 301)]:
    Wb = genW(1024, 1024, lt, sd)
    sigb = np.mean(Wb ** 2)

    q = OPQ(4, 0.45)
    q.cal(Wb)
    m1 = np.mean((Wb - q.dq(q.q(Wb))) ** 2)

    Wh2 = perch(Wb, 4)
    m2 = np.mean((Wb - Wh2) ** 2)

    Wh3, _ = dt_perch(Wb, 4)
    m3 = np.mean((Wb - Wh3) ** 2)

    snr1 = 10 * np.log10(sigb / (m1 + 1e-30))
    snr2 = 10 * np.log10(sigb / (m2 + 1e-30))
    snr3 = 10 * np.log10(sigb / (m3 + 1e-30))

    print(f"\n  {lt} (1024x1024):")
    print(f"    Global 4-bit:  SNR={snr1:.1f}dB  MSE={m1:.2e}")
    print(f"    Per-channel:   SNR={snr2:.1f}dB  MSE={m2:.2e}  (+{(1-m2/m1)*100:.0f}%)")
    print(f"    *** DT+PC:     SNR={snr3:.1f}dB  MSE={m3:.2e}  (+{(1-m3/m1)*100:.0f}%) ***")

# ============================================================
# Generate Master Figure
# ============================================================
print("\nGenerating master figure...")

fig, axes = plt.subplots(2, 3, figsize=(19, 12))

# (a) Method Ranking
ax = axes[0, 0]
sorted_r = sorted(results.items(), key=lambda x: x[1])
lbls = [s[0] for s in sorted_r]
mses = [s[1] for s in sorted_r]
rel = [m / mses[-1] for m in mses]
cols = ['#C0C0C0'] * len(lbls)
for i, l in enumerate(lbls):
    if 'DT' in l and 'Per' in l:
        cols[i] = '#70AD47'
    elif 'Per-ch' in l and 'DT' not in l:
        cols[i] = '#A5A5A5'
    elif 'Dual' in l:
        cols[i] = '#ED7D31'
bars = ax.barh(range(len(lbls)), rel, color=cols, edgecolor='white')
for bar, r, l in zip(bars, rel, lbls):
    ax.text(r + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{r:.2f}x', va='center', fontsize=9,
            fontweight='bold' if 'DT' in l else 'normal')
ax.set_yticks(range(len(lbls)))
ax.set_yticklabels([l[:40] for l in lbls], fontsize=8)
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Relative MSE (lower is better)')
ax.set_title('(a) Method Ranking (4-bit)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

# (b) Tau Search Curve
ax = axes[0, 1]
taus = np.round(np.linspace(0.005, 0.08, 16), 4)
mse_t = []
for tau in taus:
    lm = np.abs(W) >= tau
    sm = np.abs(W) < tau
    Wh = np.zeros_like(W)
    if lm.sum() > 0:
        for i in range(W.shape[0]):
            m = lm[i]
            if m.sum() > 0:
                ch = W[i, m]
                a = best_a(ch, 4)
                q = OPQ(4, a)
                q.cal(ch)
                Wh[i, m] = q.dq(q.q(ch))
    if sm.sum() > 0:
        for i in range(W.shape[0]):
            m = sm[i]
            if m.sum() > 0:
                ch = W[i, m]
                a = best_a(ch, 4)
                q = OPQ(4, a)
                q.cal(ch)
                Wh[i, m] = q.dq(q.q(ch))
    mse_t.append(np.mean((W - Wh) ** 2))

ax.plot(taus, [m / base for m in mse_t], 'b-o', markersize=5, linewidth=2)
bi = np.argmin(mse_t)
ax.annotate(f'Best tau={taus[bi]:.4f}\n{mse_t[bi]/base:.3f}x',
            xy=(taus[bi], mse_t[bi] / base),
            textcoords="offset points", xytext=(10, -15),
            fontsize=10, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Threshold tau')
ax.set_ylabel('Relative MSE')
ax.set_title('(b) Dual-Tower Tau Search', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

# (c) Per-Layer SNR
ax = axes[0, 2]
ln = list(layers.keys())
x = np.arange(len(ln))
w = 0.22
method_names = ['Global', 'Per-ch', 'DT+PC']
colors_bar = ['#C0C0C0', '#A5A5A5', '#70AD47']
for i, (lbl, col) in enumerate(zip(method_names, colors_bar)):
    ax.bar(x + i * w - w, [s[i] for s in layer_snrs.values()],
           w, label=lbl, color=col, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(ln, fontsize=9)
ax.set_ylabel('SNR (dB)')
ax.set_title('(c) Per-Layer SNR by Method', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# (d) Large-Scale 1024x1024
ax = axes[1, 0]
cats = ['attn_q\n(1024^2)', 'ffn_gate\n(1024^2)']
for idx, (lt, sd) in enumerate([('q', 300), ('g', 301)]):
    Wb = genW(1024, 1024, lt, sd)
    sigb = np.mean(Wb ** 2)
    q = OPQ(4, 0.45)
    q.cal(Wb)
    m1 = np.mean((Wb - q.dq(q.q(Wb))) ** 2)
    m2 = np.mean((Wb - perch(Wb, 4)) ** 2)
    m3 = np.mean((Wb - dt_perch(Wb, 4)[0]) ** 2)
    xpos = idx
    wd = 0.2
    ax.bar(xpos - wd, m1 / m1, wd, color='#C0C0C0', label='Global' if idx == 0 else '')
    ax.bar(xpos, m2 / m1, wd, color='#A5A5A5', label='Per-ch' if idx == 0 else '')
    ax.bar(xpos + wd, m3 / m1, wd, color='#70AD47', label='DT+PC' if idx == 0 else '')
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel('Relative MSE')
ax.set_title('(d) 1024x1024 Validation', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# (e) End-to-End SNR
ax = axes[1, 1]
el = list(e2e.keys())
es = [v[1] for v in e2e.values()]
ec = ['#C0C0C0', '#A5A5A5', '#70AD47', '#FFC000'][:len(el)]
bars = ax.barh(range(len(el)), es, color=ec, edgecolor='white')
for bar, s in zip(bars, es):
    ax.text(s + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{s:.1f}dB', va='center', fontsize=10, fontweight='bold')
ax.set_yticks(range(len(el)))
ax.set_yticklabels(el, fontsize=9)
ax.set_xlabel('E2E SNR (dB)')
ax.axvline(10, color='green', linestyle=':', alpha=0.5, label='10dB target')
ax.set_title('(e) End-to-End SNR', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# (f) Pareto Frontier
ax = axes[1, 2]
pts = [(16, 25, 'FP16'), (4, sna, '4-bit'),
       (4, snb, 'Per-ch'), (4, snc, 'DT+PC'), (3, sne, '3-bit')]
bits_p = [p[0] for p in pts]
snrp = [p[1] for p in pts]
lblp = [p[2] for p in pts]
sc = ax.scatter(bits_p, snrp,
                s=[200 if 'DT' in l else 150 for l in lblp],
                c=snrp, cmap='RdYlGn', edgecolors='black', linewidths=1, zorder=5)
for i, l in enumerate(lblp):
    ax.annotate(l, (bits_p[i], snrp[i]),
                textcoords="offset points", xytext=(8, 5),
                fontsize=9,
                fontweight='bold' if 'DT' in l else 'normal')
sp = sorted(zip(bits_p, snrp), key=lambda x: x[0])
ax.plot([s[0] for s in sp], [s[1] for s in sp], 'k--', alpha=0.3)
ax.set_xlabel('Bits/Weight')
ax.set_ylabel('SNR (dB)')
ax.set_title('(f) Storage-Accuracy Pareto', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('/data/workspace/roq_v3_master_figure.png', dpi=200, bbox_inches='tight')
print("Saved: roq_v3_master_figure.png")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL RANKING (attn_q, 4-bit)")
print("=" * 70)
for i, (nm, m) in enumerate(sorted_r):
    g = (1 - m / base) * 100
    snr_val = 10 * np.log10(sig / (m + 1e-30))
    star = " ***" if g > 30 else ""
    print(f"  #{i+1}: {nm:>40}  MSE={m:.2e}  SNR={snr_val:.2f}dB  ({g:+.1f}%){star}")

print(f"\n  BEST: DT+Per-ch -> {snc:.1f}dB end-to-end")
print("Done!")
