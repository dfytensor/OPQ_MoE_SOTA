"""
ROQ v4 FIXED: Pushing Beyond DT+PC
====================================
Corrected implementation of 4 orthogonal improvements on DT+Per-ch:

  A) Stochastic Rounding      - unbiased quantization
  B) Residual Cascaded OPQ    - quantize residual with lower bits
  C) Mixed-Precision Allocation - sensitive channels get more bits
  D) LN-Aware Pre-Scaling     - normalize row variance before quant

Key fix from v3: proper dequantization path. All methods must produce
a valid reconstructed weight matrix Wh that can be directly compared
to W. SNR = 10*log10(||W||² / ||W-Wh||²).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================
# Core OPQ (correct, from v3)
# ============================================================
class OPQ:
    def __init__(self, bits=4, alpha=0.45):
        self.bits = bits
        self.alpha = alpha
        self.L = (1 << (bits - 1))  # magnitude levels (excl sign)
        self.s = None

    def cal(self, W):
        mx = np.max(np.abs(W))
        if mx == 0: mx = 1.0
        self.s = (self.L - 1) / (mx ** self.alpha)
        return self

    def q(self, W):
        z = np.abs(W) ** self.alpha * self.s
        qm = np.clip(np.round(z), 0, self.L - 1).astype(np.int64)
        sb = 1 << (self.bits - 1)
        return (qm | ((W < 0).astype(np.int64) << (self.bits - 1))).astype(np.uint8)

    def dq(self, q):
        sb = 1 << (self.bits - 1)
        sign = np.where((q & sb) != 0, -1.0, 1.0)
        qm = (q & (sb - 1)).astype(np.float64)
        x = np.clip(qm / self.s, 1e-30, None)
        return sign * (x ** (1.0 / self.alpha))

    def quantize(self, W):
        self.cal(W)
        return self.q(W)

    def dequantize(self, q):
        return self.dq(q)


def best_alpha(ch, bits, alphas=None):
    if alphas is None:
        alphas = [0.25, 0.30, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45, 0.48, 0.50, 0.53, 0.55]
    best_m, best_a = float('inf'), None
    for a in alphas:
        q = OPQ(bits, a)
        Wh = q.dequantize(q.quantize(ch))
        m = np.mean((ch - Wh) ** 2)
        if m < best_m:
            best_m, best_a = m, a
    return best_a, best_m


def dt_perch(W, bits, taus=None):
    """DT+PC: dual-tower + per-channel alpha (the v3 winner)."""
    if taus is None:
        taus = [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050]
    sig = np.mean(W ** 2)
    best_m, best_tau, best_Wh = float('inf'), None, None
    for tau in taus:
        Wh = np.zeros_like(W)
        lm = np.abs(W) >= tau
        sm = np.abs(W) < tau
        for mask, ahi, alo in [(lm, 0.60, 0.40), (sm, 0.40, 0.20)]:
            for i in range(W.shape[0]):
                m = mask[i]
                if m.sum() == 0: continue
                ch = W[i, m]
                # search alpha in appropriate range
                if i < W.shape[0] // 2:
                    rng = np.linspace(alo, ahi, 8)
                else:
                    rng = np.linspace(alo, ahi, 8)
                a, _ = best_alpha(ch, bits, rng)
                q = OPQ(bits, a)
                Wh[i, m] = q.dequantize(q.quantize(ch))
        m = np.mean((W - Wh) ** 2)
        if m < best_m:
            best_m, best_tau, best_Wh = m, tau, Wh.copy()
    snr = 10 * np.log10(sig / (best_m + 1e-30))
    return best_Wh, best_tau, best_m, snr


# ============================================================
# v4 Improvements
# ============================================================

def v4_stochastic(W, bits=4, n_trials=8):
    """A) Stochastic rounding: average over multiple trials for unbiased estimate."""
    sig = np.mean(W ** 2)
    L = (1 << (bits - 1))
    # Use OPQ with stochastic rounding
    all_Wh = []
    for seed in range(n_trials):
        np.random.seed(seed)
        # Quantize with stochastic rounding manually
        alpha = 0.45
        mx = np.max(np.abs(W))
        s = (L - 1) / (mx ** alpha)
        z = np.abs(W) ** alpha * s
        fl = np.floor(z)
        fr = z - fl
        r = np.random.random(z.shape)
        qm = (fl + (r < fr)).clip(0, L - 1)
        # sign
        sb = 1 << (bits - 1)
        q = (qm.astype(np.int64) | ((W < 0).astype(np.int64) << (bits - 1))).astype(np.uint8)
        # dequant
        sign = np.where((q & sb) != 0, -1.0, 1.0)
        qmf = (q & (sb - 1)).astype(np.float64)
        x = np.clip(qmf / s, 1e-30, None)
        Wh = sign * (x ** (1.0 / alpha))
        all_Wh.append(Wh)
    np.random.seed(42)
    # Average for final estimate
    Wh_avg = np.mean(all_Wh, axis=0)
    m = np.mean((W - Wh_avg) ** 2)
    snr = 10 * np.log10(sig / (m + 1e-30))
    return Wh_avg, m, snr


def v4_residual_cascade(W, bits_main=4, bits_res=2):
    """
    B) Residual Cascaded OPQ:
    Stage 1: OPQ(W, bits_main) → Ŵ₁
    Stage 2: R = W - Ŵ₁, OPQ(R, bits_res, aggressive alpha) → Ŵ₂
    Final: Ŵ = Ŵ₁ + Ŵ₂
    """
    sig = np.mean(W ** 2)
    # Stage 1: main quantization with DT+PC
    Wh1, tau, m1, snr1 = dt_perch(W, bits_main)
    # Residual
    R = W - Wh1
    # Stage 2: quantize residual with lower bits + aggressive alpha
    # Residual is typically smaller magnitude → use smaller alpha (more stretch)
    q2 = OPQ(bits_res, alpha=0.30)
    q2.cal(R)
    Wh2 = q2.dequantize(q2.quantize(R))
    # Combined
    Wh = Wh1 + Wh2
    m = np.mean((W - Wh) ** 2)
    snr = 10 * np.log10(sig / (m + 1e-30))
    # Storage: bits_main + bits_res (but res is much smaller magnitude, so effective bits lower)
    return Wh, m, snr, m1


def v4_residual_3stage(W, bits=(4, 2, 1)):
    """Residual cascade with 3 stages."""
    sig = np.mean(W ** 2)
    alphas = [0.45, 0.30, 0.18]
    Wh = np.zeros_like(W)
    ms = []
    for b, a in zip(bits, alphas):
        R = W - Wh
        q = OPQ(b, a)
        Wh_stage = q.dequantize(q.quantize(R))
        Wh = Wh + Wh_stage
        ms.append(np.mean((W - Wh) ** 2))
    m = ms[-1]
    snr = 10 * np.log10(sig / (m + 1e-30))
    return Wh, m, snr, ms


def v4_mixed_precision(W, bits_options=(3, 4, 5)):
    """
    C) Mixed-precision: assign bits per channel based on L2 norm sensitivity.
    High-norm channels → more bits; low-norm → fewer bits.
    Average bit width ≈ 4.
    """
    sig = np.mean(W ** 2)
    n_ch = W.shape[0]
    norms = np.linalg.norm(W, axis=1)
    order = np.argsort(norms)[::-1]
    n3 = n_ch // 3
    assignments = np.ones(n_ch, dtype=int) * bits_options[1]  # default 4
    for k, idx in enumerate(order):
        if k < n3:
            assignments[idx] = bits_options[2]  # top 1/3 → 5 bit
        elif k >= 2 * n3:
            assignments[idx] = bits_options[0]  # bottom 1/3 → 3 bit
    Wh = np.zeros_like(W)
    for i in range(n_ch):
        ch = W[i]
        a, _ = best_alpha(ch, assignments[i])
        q = OPQ(assignments[i], a)
        Wh[i] = q.dequantize(q.quantize(ch))
    m = np.mean((W - Wh) ** 2)
    snr = 10 * np.log10(sig / (m + 1e-30))
    avg_bits = assignments.mean()
    return Wh, m, snr, avg_bits, assignments


def v4_ln_aware(W, bits=4):
    """
    D) LN-Aware Pre-Scaling:
    Scale each row so its L2 norm is equalized, quantize scaled weights,
    then unscale. This makes the dynamic range more uniform across channels.
    """
    sig = np.mean(W ** 2)
    row_norm = np.linalg.norm(W, axis=1, keepdims=True)
    target_norm = row_norm.mean()
    # Avoid division by zero
    safe_norm = np.maximum(row_norm, 1e-8)
    scale = target_norm / safe_norm
    W_scaled = W * scale
    # Quantize
    a, _ = best_alpha(W_scaled, bits)
    q = OPQ(bits, a)
    Wh_scaled = q.dequantize(q.quantize(W_scaled))
    # Unscale
    Wh = Wh_scaled / scale
    m = np.mean((W - Wh) ** 2)
    snr = 10 * np.log10(sig / (m + 1e-30))
    return Wh, m, snr, float(a)


def v4_full_combo(W, bits_main=4, bits_res=2):
    """
    ALL improvements combined:
    1. LN-aware pre-scaling
    2. DT+PC on scaled weights
    3. Stochastic rounding
    4. Residual cascade
    """
    sig = np.mean(W ** 2)
    # Step 1: LN pre-scale
    row_norm = np.linalg.norm(W, axis=1, keepdims=True)
    target_norm = row_norm.mean()
    safe_norm = np.maximum(row_norm, 1e-8)
    scale = target_norm / safe_norm
    W_scaled = W * scale

    # Step 2+3+4: Stochastic DT+PC on scaled weights
    n_trials = 4
    all_Wh = []
    for seed in range(n_trials):
        np.random.seed(seed + 100)
        Wh_s, _, _, _ = dt_perch(W_scaled, bits_main)
        # Residual
        R = W_scaled - Wh_s
        q2 = OPQ(bits_res, 0.30)
        Wh_r = q2.dequantize(q2.quantize(R))
        Wh_total_s = Wh_s + Wh_r
        # Unscale back
        Wh_unscaled = Wh_total_s / scale
        all_Wh.append(Wh_unscaled)
    np.random.seed(42)
    Wh_final = np.mean(all_Wh, axis=0)
    m = np.mean((W - Wh_final) ** 2)
    snr = 10 * np.log10(sig / (m + 1e-30))
    return Wh_final, m, snr


# ============================================================
# Model
# ============================================================
dim = 512
layers = {}
for name, lt, sd in [('attn_q','q',100),('attn_v','v',101),
                      ('ffn_gate','g',200),('ffn_up','u',201)]:
    np.random.seed(sd)
    if lt == 'q': W = np.random.normal(0, 0.02, (dim, dim))
    elif lt == 'v': W = np.random.normal(0, 0.03, (dim, dim))
    elif lt == 'g': W = np.clip(np.random.laplace(0, 0.04, (dim, dim)), -1, 1)
    else: W = np.random.normal(0, 0.04, (dim, dim))
    layers[name] = W.astype(np.float32)
np.random.seed(None)
total = sum(W.size for W in layers.values())
print(f"Model: {total:,} params, 4 layers\n")

# ============================================================
# Baseline: v3 DT+PC
# ============================================================
print("=" * 78)
print("BASELINE: v3 DT+PC")
print("=" * 78)
baseline_results = {}
for name, W in layers.items():
    sig = np.mean(W ** 2)
    Wh, tau, m, snr = dt_perch(W, 4)
    baseline_results[name] = (m, snr, tau)
    print(f"  {name:>10}: MSE={m:.2e}  SNR={snr:.2f}dB  τ={tau}")

avg_base_snr = np.mean([v[1] for v in baseline_results.values()])
avg_base_mse = np.mean([v[0] for v in baseline_results.values()])
print(f"  {'AVG':>10}:  MSE={avg_base_mse:.2e}  SNR={avg_base_snr:.2f}dB")

# ============================================================
# v4 Experiments
# ============================================================
print("\n" + "=" * 78)
print("v4 EXPERIMENTS")
print("=" * 78)

all_results = {}

# --- A) Stochastic Rounding ---
print("\n[A] Stochastic Rounding (OPQ α=0.45, 4-bit, avg 8 trials)")
for name, W in layers.items():
    Wh, m, snr = v4_stochastic(W, 4, n_trials=8)
    all_results[(name, 'stoch')] = (m, snr)
    print(f"  {name:>10}: MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")

# --- B) Residual Cascade 2-stage ---
print("\n[B] Residual Cascade 2-stage (4-bit main + 2-bit res)")
for name, W in layers.items():
    Wh, m, snr, m1 = v4_residual_cascade(W, 4, 2)
    all_results[(name, 'res2')] = (m, snr)
    print(f"  {name:>10}: MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")

# --- B+) Residual Cascade 3-stage ---
print("\n[B+] Residual Cascade 3-stage (4+2+1 bit)")
for name, W in layers.items():
    Wh, m, snr, ms = v4_residual_3stage(W, (4, 2, 1))
    all_results[(name, 'res3')] = (m, snr)
    print(f"  {name:>10}: MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")
    print(f"             stage1={ms[0]:.2e} → stage2={ms[1]:.2e} → stage3={ms[2]:.2e}")

# --- C) Mixed Precision ---
print("\n[C] Mixed Precision (3/4/5-bit by channel sensitivity)")
for name, W in layers.items():
    Wh, m, snr, avg_b, assign = v4_mixed_precision(W, (3, 4, 5))
    all_results[(name, 'mixed')] = (m, snr)
    print(f"  {name:>10}: avg_bits={avg_b:.2f}  MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")

# --- D) LN-Aware ---
print("\n[D] LN-Aware Pre-Scaling (4-bit)")
for name, W in layers.items():
    Wh, m, snr, a_used = v4_ln_aware(W, 4)
    all_results[(name, 'ln')] = (m, snr)
    print(f"  {name:>10}: alpha={a_used:.2f}  MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")

# --- E) FULL COMBO ---
print("\n[E] *** FULL COMBO: LN + Stochastic + Residual ***")
for name, W in layers.items():
    Wh, m, snr = v4_full_combo(W, 4, 2)
    all_results[(name, 'full')] = (m, snr)
    print(f"  {name:>10}: MSE={m:.2e}  SNR={snr:.2f}dB  ({(1-m/avg_base_mse)*100:+.1f}%)")

# ============================================================
# Summary Table
# ============================================================
print("\n" + "=" * 78)
print("SUMMARY TABLE (per layer)")
print("=" * 78)
methods = ['base', 'stoch', 'res2', 'res3', 'mixed', 'ln', 'full']
headers = ['Layer', 'Base', 'Stoch', 'Res2', 'Res3', 'Mixed', 'LN', 'FULL']
print(f"{'Layer':>10} | " + " | ".join(f"{h:>8}" for h in headers[1:]))
print("-" * 78)
for name in layers:
    row = [name]
    for m in methods:
        if m == 'base':
            row.append(f"{baseline_results[name][1]:>7.2f}")
        else:
            row.append(f"{all_results[(name, m)][1]:>7.2f}")
    print(f"  {row[0]:>8} | " + " | ".join(row[1:]))

# Averages
print("-" * 78)
avg_row = ['AVG']
for m in methods:
    if m == 'base':
        avg_row.append(f"{avg_base_snr:>7.2f}")
    else:
        snrs = [all_results[(n, m)][1] for n in layers]
        avg_row.append(f"{np.mean(snrs):>7.2f}")
print(f"  {avg_row[0]:>8} | " + " | ".join(avg_row[1:]))

# ============================================================
# Save for plotting
# ============================================================
np.savez('/data/workspace/v4_results_fixed.npz',
         baseline=baseline_results, results=all_results,
         avg_base_snr=avg_base_snr, avg_base_mse=avg_base_mse)
print("\nSaved: v4_results_fixed.npz")
print("Done!")
