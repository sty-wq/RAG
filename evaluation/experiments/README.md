# 实验存档（2026-11-02 ～ 11-07）

这些是 day1、day2 及之后实验的脚本、配置和汇总，从工作区 `/Users/bia/Documents/Codex/2026-11-02/wo/outputs/` 原样复制进来（2026-11-07），**脚本没有改动**。

- **能提交的**（这里）：脚本、配置、汇总 JSON / Markdown、各实验自带的 README
- **不提交的**（`evaluation/data/experiments/`，已在 .gitignore）：逐题明细、缓存、向量、HTML 报告等大文件，按同样的目录结构存放
- **没有复制的**（太大，仍在原工作区）：`hybrid-ablation-p0-p2/` 的全部 jsonl（约 540 MB）、`singlepaper-error-review-20261107/` 的 CSV 和 zip（约 126 MB）、`d-rerank-regression-audit.json/.html`、各种 `.patch`（改动已经合进代码）

⚠️ 这些脚本里写死了本机的绝对路径（比如 `/Users/bia/PycharmProjects/PythonProject5/.venv/bin/python`、原工作区的数据路径），只能在这台电脑上原样运行；要在别处复现，需要先把路径改成相对路径或参数。

| 目录 | 对应记录 | 内容 |
|---|---|---|
| `2026-11-02_day1/` | day1 | day1 文档引用的数据源哈希清单 |
| `2026-11-02_deepseek-rerank/` 等三个 | day1 §6–7 | DeepSeek 重排、排名格式修复、Qwen3 HTTP 重排的验证记录 |
| `2026-11-03_sc68-multisource-final/` | day2 §1 | 68 题答案评测（跨段综合提示词）的脚本与对照 |
| `2026-11-03_sc500-answer-review/` | day2 §1 | 500 题答案评测：Dense 前五 vs DeepSeek 重排 |
| `2026-11-03_cross16/` | day2 §2 | 16 道双论文题：归因、全文对照、一步作答、选文阅读 |
| `2026-11-03_cross93-compare/` | day2 §3 | 93 道跨论文题：一次读 50 段 vs 自动选文读正文 vs 指定全文 |
| `2026-11-03_evidence-coverage-check16/` | day2 §4 | 证据覆盖度在 16 题上的核验 |
| `2026-11-04_progress-metrics/` | — | 从 Dense 到当前版本的量化对照 |
| `2026-11-05_hybrid-ablation-p0-p2/` | — | 混合检索退化原因分析（P0–P2） |
| `2026-11-07_qwen3-reranker-baseline/` | — | Qwen3-Reranker 0.6B / 4B 在 Colab 上的基线（数据在 `evaluation/data/experiments/qwen3-reranker-baseline/`） |
| `2026-11-07_colab-experiment/` | — | Colab 实验记录 |
| `2026-11-07_sft-demo/` | — | SFT 样本构造示范（1909.00786 第一题） |
| `2026-11-07_singlepaper-error-review/` | — | 单论文错题分类审核的说明与核验（明细 CSV 未复制） |
