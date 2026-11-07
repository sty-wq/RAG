# 混合检索 P0–P2 消融

本目录为独立实验产物。没有修改项目生产代码、原索引或历史评测文件。

优先阅读 [P0–P2 结论与数据](/Users/bia/Documents/Codex/2026-11-02/wo/outputs/hybrid-ablation-p0-p2/P0-P2-混合检索退化分析.md)。

## 输入与边界

- 主实验：原始 `questions_ctx.jsonl` 中 1309 道可计分题。用于解释历史 A/B/C。
- 补充回放：改写的 500 道单论文题及已有 Dense50 缓存。用于检查现象是否只出现在旧的标题增强问题上，不能当成调参后的独立测试集。
- 语料为 416 篇论文、17887 块，沿用 `text-embedding-3-small` 索引。
- 固定每路 50 个候选、RRF 常数 60、BM25 实现与字段、证据匹配规则。
- 消融只改变 BM25 的融合权重 `alpha ∈ {0, 0.25, 0.5, 1}`，Dense 权重恒为 1。alpha=0 严格返回 Dense 原序。
- 无 reranker、无答案生成、无 LLM 评分。仅补齐原题查询向量时调用 embedding API。
- 不用金标准过滤检索。金标准只在候选冻结后计分。

## 执行

首次补齐原题 Dense 候选，在已配置 `EMBED_API_KEY` 的终端执行（支持续跑）

```bash
bash /Users/bia/Documents/Codex/2026-11-02/wo/outputs/hybrid-ablation-p0-p2/prepare.sh
```

缓存完整后，下面的分析与校验完全离线，不再调用 embedding 或 LLM

```bash
/Users/bia/PycharmProjects/PythonProject5/.venv/bin/python \
  /Users/bia/Documents/Codex/2026-11-02/wo/outputs/hybrid-ablation-p0-p2/analyze.py

/Users/bia/PycharmProjects/PythonProject5/.venv/bin/python \
  /Users/bia/Documents/Codex/2026-11-02/wo/outputs/hybrid-ablation-p0-p2/verify.py

/Users/bia/PycharmProjects/PythonProject5/.venv/bin/python \
  /Users/bia/Documents/Codex/2026-11-02/wo/outputs/hybrid-ablation-p0-p2/write_overview.py
```

`freeze.py` 已执行，不应再次执行或删除冻结目录。需要修改语料、题目、分词或候选深度时应建立新的实验目录。

## 文件说明

- `manifest.json`：题目、语料文件、代码、索引快照及原 500 题缓存的哈希。
- `frozen/`：不可变输入副本，包括原始历史结果和源代码。
- `index-working/`：供 Milvus 查询使用的独立索引副本。
- `vectors1309/`：原 1309 题本轮查询向量，可用于后续完全离线重放。
- `dense1309/`：每题的 query、Dense50 和独立 Dense10 检索结果。
- `dense500/`：复用并冻结的 500 题 Dense50。
- `bm25-original.jsonl`、`bm25-rewritten.jsonl`：每题 BM2550 的 ID、名次顺序和分数。
- `chunks.jsonl`：所有候选可通过 ID 回查的原文、标题与出处。
- `historical-audit.json`：原 A/B/C 指标重算、BM25 复现及历史 214 道退化题审计。
- `original1309/报告.md`、`rewritten500/报告.md`：P0–P2 数据表和分析。
- 各结果目录的 `details.jsonl`：逐题各配置指标、两路原分数、融合排名及证据覆盖。
- 各结果目录的 `cases.md`：正负样本的原文和排名变化。
- `original1309/reproduction-differences.json`：本轮与历史前十顺序不一致的题，保留旧新得分与指标差异。
- `verification.json`：与生产融合函数、BM25 函数的离线一致性校验。

## 如何看逐题结果

先看 `scores.dense` 与 `scores.rrf_a1`。若前者 Hit@5=1 而后者=0，检查

1. `drift.first_dense_gold_*` 查看 Dense 最靠前正确块在 BM25、融合中的名次。
2. `scores.rrf_a1.hit@50` 区分掉出前五和候选池彻底无证据。
3. `drift.wrong_top5_intruders` 查看哪些不覆盖标注证据的块从 Dense 前五之外进入融合前五。
4. `rankings` 查看 alpha=.25/.5 是否恢复该证据。
5. 用 `chunk_id` 在 `chunks.jsonl` 回查文本，结合 `evidence` 人工审阅。

相关块类别按 chunk_id 划分，证据互补性按段落编号集合划分。两者不同：两块可能覆盖同一金标准段落。
并集最多 100 条，只是召回上限，不应与最终 50 条配置当作同预算对照。
Recall 沿用历史规则，一段证据的足够长切片也可能计为覆盖；这些指标不是最终问答准确率。
