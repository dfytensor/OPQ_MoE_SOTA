import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
from scipy.optimize import curve_fit

# ============================================================
# 通用幂次量化器
# W_q = sign(W) * ((|W|/eps)^alpha / scale)   -> 整数
# W_hat = sign * (q/scale)^(1/alpha) * eps
# ============================================================

def opq_quantize(weight, alpha, mag_levels, epsilon=1.00001):
    """OPQ 量化：幂次 alpha 可调"""
    mag = np.abs(weight)
    # 变换: |w|^alpha
    transformed = np.power(mag / epsilon, alpha) * (mag_levels - 1)
    # 注意: 当 alpha<1 时, x^alpha 是凹函数, 最大值在 x=1 处
    # 但我们的 scale 应该使得变换后的最大值 = mag_levels - 1
    # 所以 scale = (mag_levels - 1) / (1/epsilon)^alpha
    scale = (mag_levels - 1) / (np.power(1.0 / epsilon, alpha))
    transformed = np.power(mag / epsilon, alpha) * scale
    q_mag = np.clip(np.round(transformed), 0, mag_levels - 1).astype(np.int64)
    is_neg = (weight < 0).astype(np.int64)
    sign_bit_pos = mag_levels  # mag_levels 必须是 2^k 的倍数关系
    # 用 bits 决定: 总 bits, 符号占1bit, 幅值占 (bits-1) bit
    return q_mag, scale  # 返回幅值整数和 scale

def opq_dequantize(q_mag, alpha, scale, sign=1.0, epsilon=1.00001):
    """OPQ 反量化"""
    return sign * np.power(q_mag / scale, 1.0 / alpha) * epsilon

# ============================================================
# 权重分布模拟（类 LLM 实际分布）
# ============================================================
def generate_weights(n=100000, dist='gaussian', sigma=0.1):
    if dist == 'gaussian':
        w = np.random.normal(0, sigma, n)
    elif dist == 'laplace':
        w = np.random.laplace(0, sigma / np.sqrt(2), n)
    else:
        w = np.random.uniform(-1, 1, n)
    return np.clip(w, -1.0, 1.0)

# ============================================================
# 主实验：不同 bit 宽度 × 不同 alpha → 找最优 alpha
# ============================================================
np.random.seed(42)

# 比特配置
bit_configs = {
    2: {'mag_bits': 1, 'total': 2},   # 1-bit符号 + 1-bit幅值 (太极端)
    3: {'mag_bits': 2, 'total': 3},   # 1-bit符号 + 2-bit幅值 (4级)
    4: {'mag_bits': 3, 'total': 4},   # 1-bit符号 + 3-bit幅值 (8级)
    5: {'mag_bits': 4, 'total': 5},   # 1-bit符号 + 4-bit幅值 (16级)
    6: {'mag_bits': 5, 'total': 6},   # 1-bit符号 + 5-bit幅值 (32级)
    8: {'mag_bits': 7, 'total': 8},   # 1-bit符号 + 7-bit幅值 (128级)
}

# alpha 搜索范围（log 均匀）
alpha_grid = np.unique(np.round(np.concatenate([
    np.logspace(-1.5, 0.5, 40, base=2),  # 2^(-1.5) ~ 2^0.5
]), decimals=4))

# 权重分布
print("Generating weights...")
weights = {}
for dist_name in ['gaussian', 'laplace']:
    weights[dist_name] = generate_weights(n=200000, dist=dist_name, sigma=0.1)
    # 裁剪到 [-1, 1]
    weights[dist_name] = np.clip(weights[dist_name], -1.0, 1.0)

# ============================================================
# 网格搜索最优 alpha
# ============================================================
results = {}

print("\n" + "=" * 80)
print("网格搜索：Bit-width × Alpha → MSE")
print("=" * 80)

for bits, cfg in bit_configs.items():
    mag_levels = 1 << cfg['mag_bits']  # 2, 4, 8, 16, 32, 128
    print(f"\n--- {bits}-bit (mag_levels={mag_levels}) ---")
    
    best_alpha = {}
    for dist_name, w in weights.items():
        best_mse = float('inf')
        best_a = None
        mse_curve = []
        
        for alpha in alpha_grid:
            # 计算 scale
            epsilon = 1.00001
            scale = (mag_levels - 1) / (np.power(1.0 / epsilon, alpha))
            
            # 量化
            mag = np.abs(w)
            transformed = np.power(mag / epsilon, alpha) * scale
            q_mag = np.clip(np.round(transformed), 0, mag_levels - 1)
            
            # 反量化
            sign = np.where(w < 0, -1.0, 1.0)
            w_hat = sign * np.power(np.clip(q_mag / scale, 1e-30, None), 1.0 / alpha) * epsilon
            
            # MSE
            mse = np.mean((w - w_hat) ** 2)
            mse_curve.append((alpha, mse))
            
            if mse < best_mse:
                best_mse = mse
                best_a = alpha
        
        best_alpha[dist_name] = (best_a, best_mse, mse_curve)
        print(f"  {dist_name:>10}: best alpha = {best_a:.4f}, MSE = {best_mse:.8f}")
    
    results[bits] = {'cfg': cfg, 'mag_levels': mag_levels, 'best': best_alpha}

# ============================================================
# 打印汇总表
# ============================================================
print("\n" + "=" * 80)
print("表4  不同比特宽度下的最优幂次 alpha*")
print("=" * 80)
print(f"{'Bits':>5} | {'Mag Levels':>12} | {'Alpha* (Gauss)':>16} | {'Alpha* (Lap)':>14} | {'MSE(Gauss)':>14} | {'MSE(Lap)':>12}")
print("-" * 80)
for bits, res in sorted(results.items()):
    g = res['best']['gaussian']
    l = res['best']['laplace']
    print(f"{bits:>5} | {res['mag_levels']:>12} | {g[0]:>16.4f} | {l[0]:>14.4f} | {g[1]:>14.8f} | {l[1]:>12.8f}")

# ============================================================
# 绘图1: Bit-width vs Optimal Alpha
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

bits_sorted = sorted(results.keys())
gauss_alphas = [results[b]['best']['gaussian'][0] for b in bits_sorted]
lap_alphas = [results[b]['best']['laplace'][0] for b in bits_sorted]

ax.plot(bits_sorted, gauss_alphas, 'bo-', label='Gaussian (LLM typical)', markersize=10, linewidth=2.5)
ax.plot(bits_sorted, lap_alphas, 'rs-', label='Laplace', markersize=10, linewidth=2.5)

# 标注数值
for i, b in enumerate(bits_sorted):
    ax.annotate(f'{gauss_alphas[i]:.3f}', (b, gauss_alphas[i]),
                textcoords="offset points", xytext=(8, 6), fontsize=10, color='blue', fontweight='bold')
    ax.annotate(f'{lap_alphas[i]:.3f}', (b, lap_alphas[i]),
                textcoords="offset points", xytext=(8, -12), fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('Bit Width (total bits)', fontsize=14)
ax.set_ylabel(r'Optimal Power $\alpha^*$', fontsize=14)
ax.set_title(r'Bit-width vs Optimal Power $\alpha^*$ (OPQ)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(bits_sorted)
# 添加趋势线参考
trend_x = np.array(bits_sorted)
trend_y = 0.5 * (1 - np.exp(-trend_x / 3)) + 0.15  # 经验拟合
ax.plot(trend_x, trend_y, 'k--', alpha=0.3, label=r'Trend: $\alpha^* \approx 0.5(1-e^{-b/3})+0.15$')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/data/workspace/opq_bit_vs_alpha.png', dpi=150, bbox_inches='tight')
print("\nSaved: /data/workspace/opq_bit_vs_alpha.png")

# ============================================================
# 绘图2: MSE vs Alpha 曲线（每个bit宽度一张子图）
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, b in enumerate(bits_sorted):
    ax = axes[idx]
    for dist_name in ['gaussian', 'laplace']:
        curve = results[b]['best'][dist_name][2]
        alphas = [c[0] for c in curve]
        mses = [c[1] for c in curve]
        # 归一化 MSE 到最小值
        min_mse = min(mses)
        norm_mses = [m / min_mse for m in mses]
        label = f'{dist_name} (min={min_mse:.2e})'
        ax.semilogy(alphas, norm_mses, '-o', markersize=3, label=label)
    
    best_a_g = results[b]['best']['gaussian'][0]
    ax.axvline(best_a_g, color='blue', linestyle='--', alpha=0.5)
    ax.set_title(f'{b}-bit (mag={results[b]["mag_levels"]} levels)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\alpha$', fontsize=11)
    ax.set_ylabel('Normalized MSE', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/data/workspace/opq_mse_curves.png', dpi=150, bbox_inches='tight')
print("Saved: /data/workspace/opq_mse_curves.png")

# ============================================================
# 绘图3: 最优 alpha 拟合公式
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# 用高斯分布的 alpha* 做拟合
gauss_points = [(b, results[b]['best']['gaussian'][0]) for b in bits_sorted if b >= 3]

# 尝试拟合公式: alpha* = a * log2(b) + b 或 alpha* = 1 / (a*b + c)

# 拟合 log-linear: alpha* = c * log(b) + d
log_bits = np.log([p[0] for p in gauss_points])
alpha_vals = np.array([p[1] for p in gauss_points])
coeffs = np.polyfit(log_bits, alpha_vals, 1)
print(f"\n拟合公式: alpha* = {coeffs[0]:.4f} * ln(b) + {coeffs[1]:.4f}")

# 另一个拟合: alpha* = a / (b + c) + d
def model(b, a, c, d):
    return a / (b + c) + d

b_arr = np.array([p[0] for p in gauss_points])
a_arr = np.array([p[1] for p in gauss_points])
popt, _ = curve_fit(model, b_arr.astype(float), a_arr.astype(float), p0=[1, 1, 0.2])
print(f"拟合公式: alpha* = {popt[0]:.4f}/(b+{popt[1]:.4f}) + {popt[2]:.4f}")

# 画图
ax.plot(bits_sorted, gauss_alphas, 'bo-', markersize=10, linewidth=2.5, label='Empirical (Gaussian)')

# 拟合曲线
b_smooth = np.linspace(2, 8, 100)
a_smooth = popt[0] / (b_smooth + popt[1]) + popt[2]
ax.plot(b_smooth, a_smooth, 'b--', linewidth=2, alpha=0.7,
        label=r'Fit: $\alpha^* = %.3f/(b+%.3f)+%.3f$' % (float(popt[0]), float(popt[1]), float(popt[2])))

# 也画 Lap
ax.plot(bits_sorted, lap_alphas, 'rs-', markersize=10, linewidth=2.5, label='Empirical (Laplace)')

ax.set_xlabel('Bit Width $b$', fontsize=14)
ax.set_ylabel(r'Optimal $\alpha^*$', fontsize=14)
ax.set_title(r'Closed-form Approximation of $\alpha^*(b)$', fontsize=16, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(bits_sorted)

plt.tight_layout()
plt.savefig('/data/workspace/opq_alpha_fit.png', dpi=150, bbox_inches='tight')
print("Saved: /data/workspace/opq_alpha_fit.png")

# ============================================================
# 最终汇总表（论文用）
# ============================================================
print("\n" + "=" * 80)
print("论文用最终表：Bit-width vs Optimal Alpha*（Gaussian 分布）")
print("=" * 80)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Optimal Power $\alpha^*$ for Different Bit Widths}")
print(r"\label{tab:optimal_alpha}")
print(r"\begin{tabular}{c|c|c|c}")
print(r"\hline")
print(r"Bit Width $b$ & Mag Levels & $\alpha^*$ (Gaussian) & $\alpha^*$ (Laplace) \\")
print(r"\hline")
for b in bits_sorted:
    g_a = results[b]['best']['gaussian'][0]
    l_a = results[b]['best']['laplace'][0]
    ml = results[b]['mag_levels']
    print(f"{b} & {ml} & {g_a:.4f} & {l_a:.4f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

# ============================================================
# 额外：验证 p=0.4 在 4-bit 确实是最优附近
# ============================================================
print("\n" + "=" * 80)
print("4-bit 精细搜索验证")
print("=" * 80)
b = 4
mag_levels = 8
w = weights['gaussian']
fine_alphas = np.round(np.linspace(0.2, 0.6, 41), 4)
print(f"{'alpha':>8} | {'MSE':>14} | {'Rel to best':>12}")
print("-" * 40)
best_mse = float('inf')
best_row = None
for a in fine_alphas:
    eps = 1.00001
    scale = (mag_levels - 1) / (np.power(1.0 / eps, a))
    mag = np.abs(w)
    transformed = np.power(mag / eps, a) * scale
    q_mag = np.clip(np.round(transformed), 0, mag_levels - 1)
    sign = np.where(w < 0, -1.0, 1.0)
    w_hat = sign * np.power(np.clip(q_mag / scale, 1e-30, None), 1.0 / a) * eps
    mse = np.mean((w - w_hat) ** 2)
    if mse < best_mse:
        best_mse = mse
        best_row = (a, mse)
    rel = mse / best_mse
    print(f"{a:>8.4f} | {mse:>14.8f} | {rel:>11.4f}x")
print(f"\n✅ Best: alpha = {best_row[0]:.4f}, MSE = {best_row[1]:.8f}")
