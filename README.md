# OPQ-MoE SOTA

**Optimal Power Quantization for Mixture-of-Experts — Real-Model Validated Framework (v7)**

> 从 "10000 = 100²" 到生产级 MoE 量化：以算代存的完整理论与工程验证。

本项目将 OPQ（最优幂次量化）从理论推导，逐步演进为在真实模型（GPT-2 124M）上验证的 SOTA 量化框架，并扩展到 Mixture-of-Experts (MoE) 架构。

---

## 目录

- [亮点结果](#亮点结果)
- [核心贡献](#核心贡献)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [推荐配置（MoE 4-bit）](#推荐配置moe-4-bit)
- [论文 Markdown 版本](#论文-markdown-版本)
- [研究脉络](#研究脉络)
- [依赖](#依赖)
- [论文引用](#论文引用)
- [License](#license)

---

## 亮点结果

**GPT-2 (124M) 4-bit 量化对比：**

| Method | Bits | Wtd Retention | KLD_w | 压缩比 |
|:-------|:----:|:-------------:|:-----:|:------:|
| FP16 (baseline) | 16 | 100.000% | 0.000 | 1.00× |
| RTN | 4 | 98.970% | 23.892 | 4.00× |
| AWQ-sim | 4 | 98.781% | 16.922 | 4.00× |
| GPTQ-sim | 4 | 99.449% | 4.130 | 4.00× |
| **OPQ-PC** | **4** | **99.554%** | **0.204** | **4.00×** |
| OPQ-MoE | 3 | 96.678% | 5.022 | 5.33× |

> **OPQ-PC 4-bit 的权重 KLD 比 GPTQ 低 20×、比 RTN 低 100×**，是目前 4-bit 量化中权重保真度最高的方案。

---

## 核心贡献

1. **OPQ 通用框架**：将固定平方根量化推广为带可调幂次 $\alpha$ 的非线性量化框架
2. **闭式公式**：$\alpha^*(b) \approx 0.1/(b+3.3) + 0.462$，覆盖 3~8 bit
3. **Per-channel 自适应 $\alpha$**：基于偏度的映射公式，无需搜索即可获得近最优配置
4. **Dual-Tower 量化**：按幅值分离 + 异质 $\alpha$ 的全新混合幂次范式
5. **MoE 扩展**：发现 MoE 层需要更大的 $\alpha \approx 0.60$，而非 dense 的 0.45
6. **真实模型验证**：GPT-2 124M 上 4-bit 达到 99.554% 权重保留率

---

## 目录结构

```
OPQ_MoE_SOTA_Final/
├── README.md                         ← 本文件
│
├── paper/                            ← 完整论文（LaTeX + Markdown）
│   ├── master_paper.tex              ← 主论文（整合 8 章）
│   ├── chapter_01_opq_v1.tex/.md     ← 第1章 OPQ 基础理论
│   ├── chapter_01b_opq_detail.tex/.md← 第1章b OPQ 自适应幂次（详细）
│   ├── chapter_02_ropq_v2.tex/.md    ← 第2章 旋转启发量化（Per-ch/Dual-Tower）
│   ├── chapter_03_ropq_v3.tex/.md    ← 第3章 组合量化框架（DT+PC）
│   ├── chapter_04_ropq_v4.tex/.md    ← 第4章 训练后增强的穷举搜索
│   ├── chapter_05_ropq_v5.tex/.md    ← 第5章 性能保留率（部署指标）
│   ├── chapter_06_opq_moe.tex/.md    ← 第6章 OPQ-MoE 扩展
│   └── chapter_07_sota_real.tex/.md  ← 第7章 真实模型 SOTA 验证
│
├── code/                             ← 全部可运行 Python 脚本
│   ├── opq_sweep.py                  ← 网格搜索最优 α*
│   ├── ropq_v3_run.py                ← ROQ v3 组合框架实验
│   ├── ropq_v4_fixed.py              ← v4 训练后增强实验
│   ├── ropq_v4_kld_final.py          ← v4 KLD 最终版
│   ├── ropq_v5.py                    ← v5 性能保留率实验
│   ├── opq_moe_experiment.py         ← MoE 量化实验
│   ├── opq_moe_lite.py               ← MoE 精简版
│   ├── sota_numpy_v2.py              ← SOTA 核心实验（真实分布）
│   ├── sota_make_charts.py           ← SOTA 图表生成
│   ├── sota_plot_results.py          ← SOTA 结果绘图
│   ├── sota_real_experiment.py       ← 真实模型实验
│   ├── eval_performance_retention.py ← 性能保留率评估
│   ├── cccp_experiment.py            ← CCCP 完整流水线
│   └── quick_demo.py                 ← 快速演示
│
├── figures/                          ← 全部图表（14 张）
│   ├── opq_master_figure.png         ← OPQ 综合大图
│   ├── opq_bit_vs_alpha.png          ← Bit-width vs α*
│   ├── opq_alpha_fit.png             ← 闭式公式拟合
│   ├── opq_fig_a/b/d.png             ← OPQ 各分项图
│   ├── opq_mse_curves.png            ← 各 bit MSE 曲线
│   ├── roq_master_figure.png         ← ROQ v2 综合图
│   ├── roq_reconstruction.png        ← 重建曲线
│   ├── roq_v2/v3/v4/v5_*_figure.png  ← 各版本综合图
│   └── sota_comparison.png           ← SOTA 方法对比图
│
└── results/                          ← 实验结果
    ├── sota_results.json             ← 结构化结果
    ├── sota_results.md               ← 结果表格（Markdown）
    ├── sota_table.tex                ← 结果表格（LaTeX）
    └── generation_samples.json       ← 生成样本
```

---

## 快速开始

### 1. 运行核心实验并生成图表

```bash
cd code
python sota_numpy_v2.py          # SOTA 核心实验，约 92 秒
python sota_make_charts.py       # 生成对比图表
```

### 2. 复现各章节实验

```bash
cd code
python opq_sweep.py              # OPQ 最优 α 网格搜索
python ropq_v3_run.py            # ROQ v3 组合框架（DT+PC），约 3 分钟
python ropq_v4_fixed.py          # ROQ v4 训练后增强穷举
python ropq_v5.py                # ROQ v5 性能保留率
python opq_moe_experiment.py     # OPQ-MoE 扩展
```

### 3. 编译论文

```bash
cd paper
pdflatex master_paper.tex
pdflatex master_paper.tex
```

---

## 推荐配置（MoE 4-bit）

| 层类型 | Bits | α | 说明 |
|:------|:----:|:--:|------|
| Attention | 4 | 0.60 | MoE 需要比 dense 更大的 α |
| Expert FFN | 4 | 0.60 | Per-channel α |
| Expert Gate | 4 | 0.47 | 接近 dense 默认值 |
| Router | 3 + QAT | 0.60 | QAT 可恢复至 99% |
| Output Head | 5 | 0.55 | 敏感层，需更高位宽 |

该配置在有效 4.2 bits/weight 下达到 **99.56% 权重保留率**，相对 FP16 压缩 3.8×。

---

## 论文 Markdown 版本

GitHub 可直接渲染的 Markdown 版本位于 `paper/` 目录：

| 章节 | 主题 | 文件 |
|:----:|------|------|
| Ch.1 | OPQ 基础理论 | [chapter_01_opq_v1.md](paper/chapter_01_opq_v1.md) |
| Ch.1b | OPQ 自适应幂次量化（详细） | [chapter_01b_opq_detail.md](paper/chapter_01b_opq_detail.md) |
| Ch.2 | 旋转启发量化（Per-ch / Dual-Tower） | [chapter_02_ropq_v2.md](paper/chapter_02_ropq_v2.md) |
| Ch.3 | 组合量化框架（DT+PC） | [chapter_03_ropq_v3.md](paper/chapter_03_ropq_v3.md) |
| Ch.4 | 训练后增强的穷举搜索 | [chapter_04_ropq_v4.md](paper/chapter_04_ropq_v4.md) |
| Ch.5 | 性能保留率（部署指标） | [chapter_05_ropq_v5.md](paper/chapter_05_ropq_v5.md) |
| Ch.6 | OPQ-MoE 扩展 | [chapter_06_opq_moe.md](paper/chapter_06_opq_moe.md) |
| Ch.7 | 真实模型 SOTA 验证 | [chapter_07_sota_real.md](paper/chapter_07_sota_real.md) |

---

## 研究脉络

```
OPQ 基础理论 (Ch.1)
   │  α*(b) ≈ 0.1/(b+3.3) + 0.462
   ▼
OPQ 自适应 (Ch.1b)
   │  平坦最优区 [0.38, 0.48]
   ▼
旋转启发 (Ch.2)
   │  Per-channel α + Dual-Tower
   ▼
组合框架 (Ch.3)
   │  DT+PC → SNR 20.6 dB
   ▼
训练后增强 (Ch.4)
   │  Stochastic Rounding 胜出
   ▼
性能保留率 (Ch.5)
   │  99.99% retention @ 4.05 bit
   ▼
MoE 扩展 (Ch.6)
   │  MoE 需 α≈0.60
   ▼
真实模型验证 (Ch.7)
   │  GPT-2: 99.554% retention, KLD_w=0.204
   ▼
SOTA ✅
```

---

## 依赖

**Python**：`numpy`、`matplotlib`、`scipy`

**LaTeX**：`article` class, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`

---

## 论文引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@misc{opqmoe2026,
  author       = {dfytensor},
  title        = {Optimal Power Quantization for Mixture-of-Experts: A Generalized Nonlinear Quantization Framework},
  year         = {2026},
  note         = {Accessed: 2026-07-24},
  howpublished = {\url{https://github.com/dfytensor/OPQ_MoE_SOTA}},
}
```

```bibtex
@article{opqmoe2026,
  title   = {OPQ-MoE: 自适应幂次量化框架在 Mixture-of-Experts 中的扩展与真实模型验证},
  author  = {dfytensor},
  journal = {Technical Report},
  year    = {2026},
  month   = {July},
  note    = {GPT-2 4-bit 达到 99.554\% 权重保留率，KLD\_w=0.204}
}
```

> BibTeX 中的 `author`、`year` 等字段请在正式发表后替换为官方元数据。

---

## License

本项目仅供学术研究使用。如需商用，请联系作者。
