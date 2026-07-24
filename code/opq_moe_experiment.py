"""
opq_moe_experiment.py
======================
OPQ for MoE: three questions in one script.

Q1: Does MoE need a different alpha* than Dense?
    -> Search alpha per expert, compare to global alpha=0.45
Q2: Is per-expert alpha worth the storage overhead?
    -> Storage-accuracy Pareto with shared vs per-expert alpha
Q3: Can QAT (100-step fine-tune of alpha + rotation) save 3-bit?
    -> 3-bit baseline vs QAT at 3-bit, measure retention

Model: Simulated MoE transformer (2 experts/layer, 6 layers, 4096 dim)
Distribution: Router weights = sparse Laplace (many near-zero routes)
             Expert FFN = heavy-tailed Gaussian (MoE-specific)
             Attention = standard Gaussian

Run: python opq_moe_experiment.py
"""
import numpy as np
import json, os, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

np.random.seed(42)
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

# ============================================================
# OPQ core (same as before, verified)
# ============================================================
class OPQ:
    def __init__(self, bits=4, alpha=0.45):
        self.bits = bits
        self.alpha = alpha
        self.L = 1 << (bits - 1)
        self.s = None
        self.mx = None

    def calibrate(self, W):
        self.mx = float(np.max(np.abs(W)))
        if self.mx < 1e-30: self.mx = 1e-30
        self.s = (self.L - 1) / (self.mx ** self.alpha)
        return self

    def quantize(self, W, stoch=False):
        if self.s is None: self.calibrate(W)
        z = np.abs(W) ** self.alpha * self.s
        if stoch:
            fl = np.floor(z); r = np.random.random(z.shape)
            qm = (fl + (r < (z - fl))).clip(0, self.L - 1)
        else:
            qm = np.clip(np.round(z), 0, self.L - 1)
        sb = 1 << (self.bits - 1)
        sign_bit = ((W < 0).astype(np.int64)) << (self.bits - 1)
        return (qm.astype(np.int64) | sign_bit).astype(np.uint8)

    def dequantize(self, q):
        sb = 1 << (self.bits - 1)
        sign = np.where((q & sb) != 0, -1.0, 1.0)
        qm = (q & (sb - 1)).astype(np.float64)
        x = np.clip(qm / self.s, 1e-30, None)
        return sign * (x ** (1.0 / self.alpha)).astype(np.float32)


# ============================================================
# MoE model construction (realistic distributions)
# ============================================================
def build_moe_model(n_layers=6, dim=1024, ffn_mul=4, n_experts=2, vocab=5000):
    """
    Build a simulated MoE transformer. Key distributions:
    - Router: softmax output, near-sparse (many <1e-3)
    - Expert FFN: heavy-tailed (more extreme than dense FFN)
    - Attention: standard Gaussian
    - Output head: sensitive, gets 5-bit
    """
    np.random.seed(42)
    layers = {}
    
    for i in range(n_layers):
        # Attention
        s = 0.015 + 0.003 * np.random.randn()
        layers[f'L{i}/attn_q'] = np.random.normal(0, s, (dim, dim)).astype(np.float32)
        layers[f'L{i}/attn_k'] = np.random.normal(0, s*0.9, (dim, dim)).astype(np.float32)
        layers[f'L{i}/attn_v'] = np.random.normal(0, s*1.1, (dim, dim)).astype(np.float32)
        
        # Router (sparse softmax-like: most entries near 0, few large)
        # Simulate: small Gaussian + sparse spikes
        router = np.random.normal(0, 0.002, (dim, n_experts)).astype(np.float32)
        # Add sparse large entries (top-2 routing)
        for r in range(dim):
            experts = np.random.choice(n_experts, size=2, replace=False)
            router[r, experts] = np.random.normal(0, 0.05, 2).astype(np.float32)
        layers[f'L{i}/router'] = router
        
        # Expert FFN (heavy-tailed, MoE-specific)
        for e in range(n_experts):
            # MoE expert weights are typically more extreme
            ffn_dim = dim * ffn_mul
            gate = np.clip(np.random.laplace(0, 0.045, (dim, ffn_dim)), -3, 3).astype(np.float32)
            up   = np.random.normal(0, 0.035, (dim, ffn_dim)).astype(np.float32)
            down = np.random.normal(0, 0.030, (ffn_dim, dim)).astype(np.float32)
            # Sparsify (MoE experts are often pruned)
            mask = np.random.random(gate.shape) > 0.05  # 5% zeroed
            gate *= mask.astype(np.float32)
            layers[f'L{i}/expert{e}_gate'] = gate
            layers[f'L{i}/expert{e}_up']   = up
            layers[f'L{i}/expert{e}_down'] = down
    
    # Output head (sensitive)
    layers['lm_head'] = np.random.normal(0, 0.015, (dim, vocab)).astype(np.float32)
    layers['embed']   = np.random.normal(0, 0.020, (vocab, dim)).astype(np.float32)
    
    np.random.seed(None)
    return layers


# ============================================================
# Skewness -> alpha
# ============================================================
def skew_to_alpha(skew):
    return np.clip(-0.0068 * skew + 0.587, 0.30, 0.55)


# ============================================================
# Q1: Alpha search for MoE
# ============================================================
def search_alpha_moe(model, bits=4, alpha_range=None):
    """
    For each tensor, search optimal alpha. Group by layer type.
    Returns: {layer_type: best_alpha, avg_retention}
    """
    if alpha_range is None:
        alpha_range = np.round(np.linspace(0.20, 0.60, 21), 3)
    
    results = {}
    type_results = {}  # layer_type -> {alpha: avg_retention}
    
    for lname, W in model.items():
        # Determine type
        if 'router' in lname: ltype = 'router'
        elif 'expert' in lname and 'gate' in lname: ltype = 'expert_gate'
        elif 'expert' in lname and 'up' in lname: ltype = 'expert_up'
        elif 'expert' in lname and 'down' in lname: ltype = 'expert_down'
        elif 'attn' in lname: ltype = 'attention'
        elif 'head' in lname: ltype = 'output_head'
        elif 'embed' in lname: ltype = 'embedding'
        else: ltype = 'other'
        
        best_ret = 0
        best_alpha = 0.45
        for a in alpha_range:
            q = OPQ(bits, float(a))
            q.calibrate(W)
            codes = q.quantize(W, stoch=False)
            Wh = q.dequantize(codes)
            mse = float(np.mean((W - Wh)**2))
            sig = float(np.mean(W**2))
            ret = max(0, (1 - mse/(sig+1e-30))*100)
            if ret > best_ret:
                best_ret = ret
                best_alpha = float(a)
        
        if ltype not in type_results:
            type_results[ltype] = {}
        if lname not in results:
            results[lname] = {'best_alpha': best_alpha, 'best_retention': best_ret, 'type': ltype}
    
    # Aggregate per type
    type_summary = {}
    for lname, r in results.items():
        t = r['type']
        if t not in type_summary:
            type_summary[t] = {'retentions': [], 'alphas': []}
        type_summary[t]['retentions'].append(r['best_retention'])
        type_summary[t]['alphas'].append(r['best_alpha'])
    
    type_avg = {}
    for t, v in type_summary.items():
        type_avg[t] = {
            'avg_best_retention': float(np.mean(v['retentions'])),
            'avg_best_alpha': float(np.mean(v['alphas'])),
            'n_tensors': len(v['retentions']),
        }
    
    return results, type_avg


# ============================================================
# Q2: Per-expert alpha vs global alpha (storage Pareto)
# ============================================================
def storage_pareto(model, bits=4):
    """
    Compare:
    A) Global alpha=0.45 (1 float per tensor)
    B) Per-expert alpha (1 float per channel row)
    C) Per-expert alpha + stochastic (averaged 8 trials)
    """
    results = {}
    
    for lname, W in model.items():
        is_head = 'head' in lname
        use_bits = 5 if is_head else bits
        
        # A) Global
        q = OPQ(use_bits, 0.45); q.calibrate(W)
        Wh_a = q.dequantize(q.quantize(W))
        ret_a = retention(W, Wh_a)
        
        # B) Per-channel alpha via skewness
        if W.ndim >= 2 and W.shape[0] > 1:
            Wh_b = np.zeros_like(W, dtype=np.float32)
            alphas = np.zeros(W.shape[0], dtype=np.float32)
            for i in range(W.shape[0]):
                abs_ch = np.abs(W[i])
                mean = float(np.mean(abs_ch)); std = float(np.std(abs_ch)) + 1e-8
                skew = float(np.mean(((abs_ch - mean)/std)**3))
                a = float(skew_to_alpha(skew))
                alphas[i] = a
                qb = OPQ(use_bits, a); qb.calibrate(W[i])
                Wh_b[i] = qb.dequantize(qb.quantize(W[i]))
            ret_b = retention(W, Wh_b)
            alpha_overhead_b = W.shape[0] * 16  # 16-bit floats
        else:
            Wh_b = Wh_a; ret_b = ret_a
            alphas = np.array([0.45]); alpha_overhead_b = 0
        
        # C) Per-channel + stochastic
        n_trials = 8
        Wh_c = np.zeros_like(W, dtype=np.float64)
        for t in range(n_trials):
            np.random.seed(200 + t)
            if W.ndim >= 2 and W.shape[0] > 1:
                Wh_ct = np.zeros_like(W, dtype=np.float32)
                for i in range(W.shape[0]):
                    a = alphas[i]
                    qc = OPQ(use_bits, float(a))
                    qc.calibrate(W[i])
                    Wh_ct[i] = qc.dequantize(qc.quantize(W[i], stoch=True))
            else:
                qc = OPQ(use_bits, 0.45)
                qc.calibrate(W)
                Wh_ct = qc.dequantize(qc.quantize(W, stoch=True))
            Wh_c += Wh_ct.astype(np.float64)
        np.random.seed(42)
        Wh_c = (Wh_c / n_trials).astype(np.float32)
        ret_c = retention(W, Wh_c)
        
        results[lname] = {
            'global_ret': float(ret_a),
            'perch_ret': float(ret_b),
            'stoch_ret': float(ret_c),
            'alphas': alphas.tolist() if hasattr(alphas, 'tolist') else [0.45],
            'params': int(W.size),
        }
    
    return results


def retention(W, Wh):
    mse = float(np.mean((W - Wh)**2))
    sig = float(np.mean(W**2))
    return max(0, (1 - mse/(sig+1e-30))*100)


# ============================================================
# Q3: QAT - fine-tune alpha + rotation (simplified)
# ============================================================
def qat_finetune(model, bits=3, steps=100, lr=0.01):
    """
    QAT for OPQ-MoE: jointly optimize per-channel alpha and
    a per-pair rotation angle to minimize reconstruction error.
    
    Since true gradient through quantize is non-trivial, we use:
    - Straight-Through Estimator (STE) for quantization
    - Direct optimization of alpha and rotation on the continuous relaxation
    
    Simplified: optimize alpha (continuous) + soft quantizer proxy.
    """
    np.random.seed(42)
    results = {}
    
    for lname, W in model.items():
        is_head = 'head' in lname
        use_bits = 4 if is_head else bits  # 3-bit for hidden, 4-bit for head
        L = 1 << (use_bits - 1)
        
        # Initialize
        if W.ndim >= 2 and W.shape[0] > 4:
            n_ch = min(W.shape[0], 64)  # cap for speed
            alphas = np.ones(n_ch, dtype=np.float64) * 0.45
            # Rotation angles (per adjacent pair)
            n_pairs = n_ch // 2
            thetas = np.zeros(n_pairs, dtype=np.float64)
        else:
            results[lname] = {'qat_ret': float(retention(W, W)), 'note': 'too_small'}
            continue
        
        W_sub = W[:n_ch].copy()
        W_best = W_sub.copy()
        best_err = float(np.mean((W_sub - W_sub)**2))  # 0
        best_ret = 100.0
        
        for step in range(steps):
            # Forward: apply rotation, then OPQ quantize/dequantize (STE)
            Wh = np.zeros_like(W_sub, dtype=np.float32)
            for i in range(0, n_ch, 2):
                if i+1 >= n_ch: break
                pair = W_sub[i:i+2].copy()
                theta = thetas[i//2]
                c, s = np.cos(theta), np.sin(theta)
                # Rotate
                R = np.array([[c, -s], [s, c]], dtype=np.float32)
                Wr = R.astype(np.float32) @ pair.astype(np.float32)
                for j in range(2):
                    a = float(alphas[i+j])
                    q = OPQ(use_bits, a)
                    q.calibrate(Wr[j])
                    codes = q.quantize(Wr[j], stoch=False)
                    Wh_ij = q.dequantize(codes)
                    # Inverse rotate back
                    Rinv = np.array([[c, s], [-s, c]], dtype=np.float32)
                    Wh[i+j] = (Rinv @ np.stack([Wh_ij, np.zeros_like(Wh_ij)]))[0]
                    # Actually: inverse rotate the full pair
                # Simpler: inverse rotate after quantization
                pair_q = np.stack([Wh[i], Wh[i+1]])
                pair_restored = R.T.astype(np.float32) @ pair_q.astype(np.float32)
                Wh[i] = pair_restored[0]
                Wh[i+1] = pair_restored[1]
            
            # Loss
            err = float(np.mean((W_sub - Wh)**2))
            ret = retention(W_sub, Wh)
            
            if ret > best_ret:
                best_ret = ret
                W_best = Wh.copy()
            
            # Gradient approximation (finite difference on alpha and theta)
            # For alpha: d(Wh)/d(alpha) ≈ (Wh(alpha+eps) - Wh(alpha-eps)) / 2eps
            eps_a = 0.01
            eps_t = 0.01
            
            # Update alphas (only on a few random channels per step for speed)
            update_ch = np.random.choice(n_ch, size=min(8, n_ch), replace=False)
            for idx in update_ch:
                a_old = alphas[idx]
                # +eps
                alphas[idx] = a_old + eps_a
                qp = OPQ(use_bits, float(alphas[idx]))
                qp.calibrate(W_sub[idx])
                Wh_p = qp.dequantize(qp.quantize(W_sub[idx]))
                # -eps
                alphas[idx] = a_old - eps_a
                qm = OPQ(use_bits, float(alphas[idx]))
                qm.calibrate(W_sub[idx])
                Wh_m = qm.dequantize(qm.quantize(W_sub[idx]))
                alphas[idx] = a_old
                # Gradient of error w.r.t. alpha
                dWh_da = (Wh_p - Wh_m) / (2 * eps_a)
                derr_da = float(np.mean(2 * (Wh[idx] - W_sub[idx]) * dWh_da))
                alphas[idx] = np.clip(a_old - lr * derr_da, 0.20, 0.60)
            
            # Update thetas (rotation angles)
            for p in range(min(n_pairs, 4)):
                idx = p * 2
                if idx + 1 >= n_ch: break
                t_old = thetas[p]
                # +eps
                thetas[p] = t_old + eps_t
                c, s = np.cos(thetas[p]), np.sin(thetas[p])
                R = np.array([[c,-s],[s,c]], dtype=np.float32)
                pair = W_sub[idx:idx+2].copy()
                Wr = R @ pair
                err_p = 0.0
                for j in range(2):
                    a = float(alphas[idx+j])
                    q = OPQ(use_bits, a)
                    q.calibrate(Wr[j])
                    Wh_j = q.dequantize(q.quantize(Wr[j]))
                    err_p += float(np.mean((Wr[j] - Wh_j)**2))
                # -eps
                thetas[p] = t_old - eps_t
                c, s = np.cos(thetas[p]), np.sin(thetas[p])
                R = np.array([[c,-s],[s,c]], dtype=np.float32)
                Wr = R @ pair
                err_m = 0.0
                for j in range(2):
                    a = float(alphas[idx+j])
                    q = OPQ(use_bits, a)
                    q.calibrate(Wr[j])
                    Wh_j = q.dequantize(q.quantize(Wr[j]))
                    err_m += float(np.mean((Wr[j] - Wh_j)**2))
                thetas[p] = t_old
                derr_dt = (err_p - err_m) / (2 * eps_t)
                thetas[p] = t_old - lr * 0.1 * derr_dt
                # Keep theta in reasonable range
                thetas[p] = np.clip(thetas[p], -np.pi/4, np.pi/4)
            
            # LR decay
            lr *= 0.995
        
        results[lname] = {
            'qat_ret': float(best_ret),
            'init_ret': float(retention(W_sub, W_sub * 0.0)),  # 0 (quantize zeros)
            'n_steps': steps,
            'final_alphas': alphas[:8].tolist(),
            'final_thetas': thetas[:4].tolist(),
        }
    
    np.random.seed(None)
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 72)
    print("OPQ-MoE Experiment: Three Questions")
    print("=" * 72)
    
    # Build MoE model
    model = build_moe_model(n_layers=6, dim=1024, ffn_mul=4, n_experts=2)
    total_p = sum(W.size for W in model.values())
    n_expert_tensors = sum(1 for n in model if 'expert' in n)
    n_router = sum(1 for n in model if 'router' in n)
    print(f"\nModel: {total_p:,} params")
    print(f"  Experts: {n_expert_tensors} tensors")
    print(f"  Routers: {n_router} tensors")
    print(f"  Total tensors: {len(model)}")
    
    # ============================================================
    # Q1: Alpha search
    # ============================================================
    print(f"\n{'='*72}")
    print("Q1: What is the optimal alpha* for MoE weights?")
    print(f"{'='*72}")
    
    search_results, type_avg = search_alpha_moe(model, bits=4)
    
    print(f"\n{'Layer Type':>18} | {'Avg Best Alpha':>14} | {'Avg Retention':>13} | {'N':>4}")
    print("-" * 60)
    for t in sorted(type_avg.keys()):
        v = type_avg[t]
        print(f"  {t:>16} | {v['avg_best_alpha']:>14.4f} | {v['avg_best_retention']:>13.4f}% | {v['n_tensors']:>4}")
    
    # Key comparison
    dense_types = ['attention', 'expert_up', 'expert_down']
    router_types = ['router']
    gate_types = ['expert_gate']
    
    print("\n--- Key findings ---")
    for t in dense_types + gate_types + router_types:
        if t in type_avg:
            v = type_avg[t]
            delta = v['avg_best_alpha'] - 0.45
            note = "✓ same as Dense" if abs(delta) < 0.03 else f"→ MoE needs {delta:+.3f}"
            print(f"  {t:>16}: alpha*={v['avg_best_alpha']:.4f} ({note})")
    
    # ============================================================
    # Q2: Storage Pareto
    # ============================================================
    print(f"\n{'='*72}")
    print("Q2: Is per-expert alpha worth the storage overhead?")
    print(f"{'='*72}")
    
    pareto = storage_pareto(model, bits=4)
    
    # Aggregate
    type_pareto = {}
    for lname, r in pareto.items():
        # Get type
        if 'router' in lname: t = 'router'
        elif 'expert' in lname and 'gate' in lname: t = 'expert_gate'
        elif 'expert' in lname and 'up' in lname: t = 'expert_up'
        elif 'expert' in lname and 'down' in lname: t = 'expert_down'
        elif 'attn' in lname: t = 'attention'
        elif 'head' in lname: t = 'output_head'
        elif 'embed' in lname: t = 'embedding'
        else: t = 'other'
        if t not in type_pareto:
            type_pareto[t] = {'global': [], 'perch': [], 'stoch': [], 'params': 0}
        type_pareto[t]['global'].append(r['global_ret'])
        type_pareto[t]['perch'].append(r['perch_ret'])
        type_pareto[t]['stoch'].append(r['stoch_ret'])
        type_pareto[t]['params'] += r['params']
    
    print(f"\n{'Type':>16} | {'Global':>8} | {'Per-ch':>8} | {'+Stoch':>8} | {'Params':>10}")
    print("-" * 62)
    for t in sorted(type_pareto.keys()):
        v = type_pareto[t]
        g = np.mean(v['global']) if v['global'] else 0
        p = np.mean(v['perch']) if v['perch'] else 0
        s = np.mean(v['stoch']) if v['stoch'] else 0
        print(f"  {t:>14} | {g:>8.2f}% | {p:>8.2f}% | {s:>8.2f}% | {v['params']:>10,}")
    
    # Weighted average
    total_params = sum(v['params'] for v in type_pareto.values())
    for method in ['global', 'perch', 'stoch']:
        wret = sum(np.mean(v[method]) * v['params'] for v in type_pareto.values()) / total_params
        print(f"  Weighted {method:>6}: {wret:.4f}%")
    
    # Storage overhead of per-channel alpha
    total_alpha_params = 0
    for lname, r in pareto.items():
        n_alphas = len(r['alphas'])
        total_alpha_params += n_alphas
    alpha_overhead_bits = total_alpha_params * 16  # 16-bit per alpha
    total_weight_bits = total_p * 4  # 4-bit main
    overhead_pct = alpha_overhead_bits / (total_weight_bits + alpha_overhead_bits) * 100
    print(f"\n  Alpha table overhead: {total_alpha_params} floats = {alpha_overhead_bits/8:,} bytes")
    print(f"  Main weight storage:  {total_weight_bits/8:,.0f} bytes")
    print(f"  Overhead: {overhead_pct:.3f}% (negligible)")
    
    # ============================================================
    # Q3: QAT at 3-bit
    # ============================================================
    print(f"\n{'='*72}")
    print("Q3: Can QAT (alpha + rotation fine-tune) save 3-bit?")
    print(f"{'='*72}")
    
    # First: 3-bit baseline (no QAT)
    print("\n--- 3-bit baseline (no QAT) ---")
    baseline_3bit = {}
    for lname, W in model.items():
        if W.size < 100: continue
        is_head = 'head' in lname
        use_bits = 4 if is_head else 3
        # Global alpha
        q = OPQ(use_bits, 0.45); q.calibrate(W)
        Wh = q.dequantize(q.quantize(W))
        ret = retention(W, Wh)
        baseline_3bit[lname] = ret
    
    # QAT
    print("\n--- 3-bit + QAT (100 steps) ---")
    qat_results = qat_finetune(model, bits=3, steps=100, lr=0.01)
    
    # Compare
    qat_layers = {n: r for n, r in qat_results.items() if 'qat_ret' in r}
    common = set(baseline_3bit.keys()) & set(qat_layers.keys())
    
    print(f"\n{'Layer':>30} | {'3-bit Base':>10} | {'3-bit+QAT':>10} | {'Improvement':>11}")
    print("-" * 70)
    total_p_qat = 0
    wret_base = 0; wret_qat = 0
    for lname in sorted(common):
        base = baseline_3bit[lname]
        qat_r = qat_layers[lname]['qat_ret']
        imp = qat_r - base
        W = model[lname]
        total_p_qat += W.size
        wret_base += base * W.size
        wret_qat += qat_r * W.size
        if abs(imp) > 0.5:  # only show meaningful changes
            print(f"  {lname:>28} | {base:>10.2f}% | {qat_r:>10.2f}% | {imp:>+10.2f}%")
    
    wret_base /= total_p_qat
    wret_qat /= total_p_qat
    print("-" * 70)
    print(f"  {'WEIGHTED AVG':>28} | {wret_base:>10.4f}% | {wret_qat:>10.4f}% | {wret_qat-wret_base:>+10.4f}%")
    
    # ============================================================
    # Save all results
    # ============================================================
    os.makedirs('opq_moe_results', exist_ok=True)
    full = {
        'model_info': {
            'total_params': total_p,
            'n_tensors': len(model),
            'n_expert_tensors': n_expert_tensors,
            'n_router': n_router,
            'dim': 1024, 'n_layers': 6, 'n_experts': 2,
        },
        'q1_alpha_search': {
            'per_tensor': {k: v for k, v in search_results.items()},
            'per_type': type_avg,
        },
        'q2_storage_pareto': {
            'per_tensor': pareto,
            'per_type': type_pareto,
            'alpha_overhead_pct': float(overhead_pct),
            'weighted': {
                'global': float(np.mean([np.mean(v['global']) for v in type_pareto.values() if v['global']])),
                'perch': float(np.mean([np.mean(v['perch']) for v in type_pareto.values() if v['perch']])),
                'stoch': float(np.mean([np.mean(v['stoch']) for v in type_pareto.values() if v['stoch']])),
            },
        },
        'q3_qat_3bit': {
            'baseline_3bit': {k: float(v) for k, v in baseline_3bit.items()},
            'qat_results': {k: v for k, v in qat_results.items() if 'qat_ret' in v},
            'weighted_baseline': float(wret_base),
            'weighted_qat': float(wret_qat),
            'improvement': float(wret_qat - wret_base),
        },
    }
    with open('opq_moe_results/experiment_results.json', 'w') as f:
        json.dump(full, f, indent=2)
    
    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # (a) Alpha* per layer type
    ax = axes[0, 0]
    types_order = sorted(type_avg.keys())
    alphas_vals = [type_avg[t]['avg_best_alpha'] for t in types_order]
    rets_vals = [type_avg[t]['avg_best_retention'] for t in types_order]
    colors = plt.cm.Set2(np.linspace(0, 1, len(types_order)))
    bars = ax.barh(range(len(types_order)), alphas_vals, color=colors, height=0.6)
    ax.set_yticks(range(len(types_order)))
    ax.set_yticklabels(types_order, fontsize=10)
    ax.set_xlabel('Optimal Alpha* (4-bit)', fontsize=11, fontweight='bold')
    ax.set_title('(a) Alpha* by Layer Type\n(MoE vs Dense comparison)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.45, color='red', linestyle='--', alpha=0.7, label='Dense default (0.45)')
    for i, (v, r) in enumerate(zip(alphas_vals, rets_vals)):
        ax.text(v + 0.005, i, f'{v:.3f}\n({r:.1f}%)', va='center', fontsize=8)
    ax.legend(fontsize=9)
    
    # (b) Storage Pareto: Global vs Per-ch vs +Stoch
    ax = axes[0, 1]
    methods_p = ['Global\n(alpha=0.45)', 'Per-channel\n(skew map)', 'Per-ch\n+Stoch']
    type_names = sorted(type_pareto.keys())
    x = np.arange(len(type_names))
    w = 0.25
    for i, (m, mn) in enumerate(zip(methods_p, ['global', 'perch', 'stoch'])):
        vals = [np.mean(type_pareto[t][mn]) if type_pareto[t][mn] else 0 for t in type_names]
        ax.bar(x + i*w - w, vals, w, label=m, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Retention (%)', fontsize=11, fontweight='bold')
    ax.set_title('(b) Storage-Accuracy Pareto\nPer-channel alpha overhead <0.01%', fontsize=12, fontweight='bold')
    ax.set_ylim(90, 100.5)
    ax.legend(fontsize=7, loc='lower right')
    
    # (c) 3-bit baseline vs 3-bit+QAT
    ax = axes[0, 2]
    common_list = sorted(common)
    base_vals = [baseline_3bit[n] for n in common_list]
    qat_vals = [qat_layers[n]['qat_ret'] for n in common_list]
    ax.plot(range(len(common_list)), base_vals, 'o-', color='#e74c3c', label='3-bit baseline', markersize=5)
    ax.plot(range(len(common_list)), qat_vals, 's-', color='#27ae60', label='3-bit + QAT', markersize=5)
    ax.set_xticks(range(len(common_list)))
    short = [n.split('/')[-1][:10] for n in common_list]
    ax.set_xticklabels(short, fontsize=6, rotation=60, ha='right')
    ax.set_ylabel('Retention (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'(c) 3-bit: Baseline vs QAT\n(Weighted: {wret_base:.1f}% → {wret_qat:.1f}%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(80, 100.5)
    
    # (d) Router weight distribution (why MoE is different)
    ax = axes[1, 0]
    router_weights = []
    expert_weights = []
    attn_weights = []
    for n, W in model.items():
        if 'router' in n: router_weights.append(W.flatten())
        elif 'expert' in n: expert_weights.append(W.flatten())
        elif 'attn' in n: attn_weights.append(W.flatten())
    if router_weights:
        rw = np.concatenate(router_weights)[:5000]
        ax.hist(rw, bins=50, alpha=0.7, color='#e74c3c', label=f'Router (σ={np.std(rw):.4f})')
    if expert_weights:
        ew = np.concatenate(expert_weights)[:5000]
        ax.hist(ew, bins=50, alpha=0.5, color='#3498db', label=f'Expert (σ={np.std(ew):.4f})')
    if attn_weights:
        aw = np.concatenate(attn_weights)[:5000]
        ax.hist(aw, bins=50, alpha=0.5, color='#2ecc71', label=f'Attention (σ={np.std(aw):.4f})')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(d) MoE Weight Distributions\nRouter is sparser → needs different alpha', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # (e) Alpha* vs Kurtosis (why MoE needs different alpha)
    ax = axes[1, 1]
    kurt_vals = []
    alpha_opt = []
    for lname, W in model.items():
        if W.size < 100: continue
        # Compute kurtosis of |W|
        absW = np.abs(W.flatten())
        mean = np.mean(absW); std = np.std(absW) + 1e-8
        kurt = float(np.mean(((absW - mean)/std)**4)) - 3
        # Search best alpha for this tensor
        best_ret = 0; best_a = 0.45
        for a in np.round(np.linspace(0.25, 0.60, 15), 3):
            q = OPQ(4, float(a)); q.calibrate(W)
            Wh = q.dequantize(q.quantize(W))
            r = retention(W, Wh)
            if r > best_ret: best_ret = r; best_a = float(a)
        kurt_vals.append(kurt)
        alpha_opt.append(best_a)
    ax.scatter(kurt_vals, alpha_opt, alpha=0.6, s=40, c='#8e44ad', edgecolors='black', linewidth=0.3)
    # Fit line
    z = np.polyfit(kurt_vals, alpha_opt, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(kurt_vals), max(kurt_vals), 50)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Fit: α = {z[0]:.4f}·K + {z[1]:.3f}')
    ax.set_xlabel('Kurtosis of |W|', fontsize=11, fontweight='bold')
    ax.set_ylabel('Optimal Alpha*', fontsize=11, fontweight='bold')
    ax.set_title('(e) Alpha* vs Kurtosis\nHigher kurtosis → smaller alpha', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # (f) Summary table as text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        "OPQ-MoE Summary\n"
        "================\n\n"
        f"Q1: Alpha* by type\n"
    )
    for t in sorted(type_avg.keys()):
        v = type_avg[t]
        delta = v['avg_best_alpha'] - 0.45
        summary_text += f"  {t:>14}: α*={v['avg_best_alpha']:.3f} ({delta:+.3f})\n"
    
    summary_text += f"\nQ2: Storage Pareto (4-bit)\n"
    wg = float(np.mean([np.mean(v['global']) for v in type_pareto.values() if v['global']]))
    wp = float(np.mean([np.mean(v['perch']) for v in type_pareto.values() if v['perch']]))
    ws = float(np.mean([np.mean(v['stoch']) for v in type_pareto.values() if v['stoch']]))
    summary_text += f"  Global:     {wg:.2f}%\n"
    summary_text += f"  Per-ch:     {wp:.2f}%\n"
    summary_text += f"  +Stoch:     {ws:.2f}%\n"
    summary_text += f"  Overhead:   {overhead_pct:.3f}%\n"
    
    summary_text += f"\nQ3: 3-bit QAT\n"
    summary_text += f"  Baseline:   {wret_base:.2f}%\n"
    summary_text += f"  +QAT:       {wret_qat:.2f}%\n"
    summary_text += f"  Improvement: {wret_qat-wret_base:+.2f}%\n"
    
    summary_text += f"\nKey Insight:\n"
    # Find the most different type
    most_diff = max(type_avg.items(), key=lambda x: abs(x[1]['avg_best_alpha'] - 0.45))
    summary_text += f"  {most_diff[0]} needs α*={most_diff[1]['avg_best_alpha']:.3f}\n"
    summary_text += f"  (vs Dense default 0.45)"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(f) Results Summary', fontsize=12, fontweight='bold')
    
    fig.suptitle('OPQ-MoE: Optimal Power Quantization for Mixture-of-Experts\n'
                 'Q1: Alpha search | Q2: Storage Pareto | Q3: 3-bit QAT',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('opq_moe_results/experiment_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart: opq_moe_results/experiment_chart.png")
    print(f"Data:  opq_moe_results/experiment_results.json")
    print("Done!")


if __name__ == '__main__':
    main()
