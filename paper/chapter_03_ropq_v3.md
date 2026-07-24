# Chapter 3: ROQ v3 — Composite Quantization Framework

> 终极组合：验证 "4-bit ROQ v3 ≈ FP16" 的核心假设

---

## 目录

- [1. Overview](#1-overview)
- [2. Composite Framework](#2-composite-framework)
- [3. Core Results](#3-core-results)
- [4. Theoretical Analysis](#4-theoretical-analysis)
- [5. Reproducibility](#5-reproducibility)

---

## 1. Overview

本章将前几章的发现组合为统一的量化框架，验证 "4-bit ROQ v3 $\approx$ FP16" 的核心假设。最终方案 **DT+PC (Dual-Tower + Per-Channel)** 在 attention_q 层上将 4-bit 量化 SNR 从 15.15 dB 提升至 **20.61 dB（+71.6%）**，端到端 SNR 从 8.30 dB 提升至 **13.79 dB（+66%）**。

---

## 2. Composite Framework

完整的 ROQ v3 流水线如下：

1. **旋转预变换**（可选，当前实验未启用以避免梯度问题）：对权重 $W$ 左乘块对角正交矩阵 $R$，将通道间相关性降到最低。
2. **双塔分离**：按幅值阈值 $\tau$ 拆分为大值塔和小值塔。
3. **Per-channel 幂次适配**：每个通道独立搜索最优 $\alpha$。
4. **混合精度分配**：按敏感度给每层分配不同 bit 宽度。

---

## 3. Core Results

### 3.1 Weight-level MSE (attn_q, 4-bit)

**表 1：4-bit 方法排名（attention_q 层, dim=512）**

| Rank | Method | MSE | SNR (dB) | Gain |
|:----:|:-------|:---:|:--------:|:----:|
| **1** | **DT+Per-ch $\tau^*$=0.030** | **3.46e-6** | **20.61** | **+71.6%** |
| 2 | Per-channel $\alpha$ search | 6.21e-6 | 18.08 | +49.1% |
| 3 | Dual-Tower (global $\alpha$) | 1.12e-5 | 15.52 | +8.3% |
| 4 | Global OPQ $\alpha$=0.50 | 1.15e-5 | 15.39 | +5.5% |
| 5 | Global OPQ $\alpha$=0.45 (baseline) | 1.22e-5 | 15.15 | 0.0% |
| 6 | Global 3-bit $\alpha$=0.40 | 7.55e-5 | 7.22 | -520% |

### 3.2 End-to-End SNR (full Transformer block)

**表 2：端到端推理 SNR（4 层 Transformer block, dim=512）**

| Rank | Method | E2E SNR (dB) |
|:----:|:-------|:------------:|
| **1** | **DT+Per-ch 4-bit** | **13.79** |
| 2 | Per-channel $\alpha$ 4-bit | 11.43 |
| 3 | Global 4-bit OPQ | 8.30 |
| 4 | Global 3-bit | -2.20 |

### 3.3 Per-layer Breakdown

**表 3：各层最佳方法（4-bit）**

| Layer | Best Method | Best SNR | 2nd Best |
|:------|:------------|:--------:|:--------:|
| attn_q | DT+PC | 20.6 dB | Per-ch (18.1) |
| attn_v | DT+PC | 20.6 dB | Per-ch (17.5) |
| ffn_gate | DT+PC | 18.7 dB | Per-ch (16.2) |
| ffn_up | DT+PC | 20.3 dB | Per-ch (17.8) |

> 关键观察：**DT+PC 在所有层上都是最优方法**，且优势一致（+50% 到 +74%）。

### 3.4 Large-Scale Validation (1024×1024)

**表 4：大规模验证（1024×1024 权重矩阵）**

| Layer | Method | SNR (dB) | Gain |
|:------|:-------|:--------:|:----:|
| attn_q | Global 4-bit | 15.3 | 0% |
|  | Per-channel | 17.7 | +44% |
|  | DT+Per-ch | **20.3** | **+69%** |
| ffn_gate | Global 4-bit | 12.7 | 0% |
|  | Per-channel | 16.2 | +54% |
|  | DT+PC | **18.2** | **+72%** |

---

## 4. Theoretical Analysis

### 4.1 Why DT+PC Works

双塔分离的本质是 **对不同幅值分布使用不同的幂次变换**：

- **大值（$\geq \tau$）**：分布较平坦，适合 $\alpha \approx 0.50$（Sqrt），因为大值的相对误差对 SNR 贡献大，Sqrt 在大值区近似线性，保精度好。
- **小值（$< \tau$）**：分布尖锐且集中在零附近，适合 $\alpha \approx 0.25 \sim 0.40$（高次根），拉伸小值区的动态范围，和 OPQ 的核心思想一致。

Per-channel $\alpha$ 搜索进一步优化：每个通道的分布偏度不同，最优 $\alpha$ 在 0.30（尖峰通道）到 0.60（平坦通道）之间连续变化。

### 4.2 Threshold Selection

实验发现最优阈值 $\tau^* \approx 0.02 \sim 0.03$（对 std$\approx$0.02 的权重），对应约 25%~35% 的权重属于"大值塔"。这个比例使得两塔的量化误差大致平衡，避免某一塔成为瓶颈。

### 4.3 Storage-Accuracy Pareto

DT+PC 的存储开销分析：

- 权重量化：4 bit/权重（两塔都用 4-bit）
- Tower 标记位：1 bit/权重（标识属于哪个塔）
- Per-channel $\alpha$ 存储：每通道 8-bit（128 个通道 × 8-bit = 128 B，可忽略）
- Scale 存储：每通道 16-bit（同样可忽略）

**有效比特宽度 = 5 bit/权重**（4-bit 量化 + 1-bit 塔标记）。相比 FP16 的 16-bit，压缩比 3.2×，SNR 从理论最大约 25 dB 降到 20.6 dB（仅损失 4.4 dB）。

---

## 5. Reproducibility

全部实验代码在 `code/ropq_v3_run.py` 中，约 360 行。主要依赖：numpy, matplotlib。在 CPU 上完整运行约 3 分钟（dim=512, 4 层）。图表自动生成在 `figures/roq_v3_master_figure.png`。
