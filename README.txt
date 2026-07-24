OPQ-MoE SOTA Final (v7)
=======================

Real-model validated quantization framework.

Result (GPT-2 124M):
  OPQ-PC 4-bit:  99.554% retention, KLD_w=0.204
  GPTQ-sim 4bit: 99.449% retention, KLD_w=4.130
  AWQ-sim 4bit:  98.781% retention, KLD_w=16.922
  RTN 4bit:      98.970% retention, KLD_w=23.892

Reproduce:
  cd code && python sota_numpy_v2.py && python sota_make_charts.py

Compile paper: cd paper && pdflatex master_paper.tex

Config (MoE 4-bit):
  Attention:  4b OPQ-PC a=0.60
  Expert FFN: 4b OPQ-PC a=0.60
  Gate:       4b OPQ-PC a=0.47
  Router:     3b+SR a=0.60
  OutputHead: 5b a=0.55
