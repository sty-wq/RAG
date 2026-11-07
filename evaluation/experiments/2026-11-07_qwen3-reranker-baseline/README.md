# Qwen3-Reranker 0.6B / 4B 基准测试

目的：在同一批 Dense Top50 候选上，比较两个开源模型的排序效果、速度与显存。这里只跑微调前基准，不训练，不生成答案，不调用 embedding 或 DeepSeek，不连接 Milvus。

## 服务器执行

把 `qwen3-reranker-baseline.tar.gz` 上传到服务器 `/workspace`。在已经通过 GPU 检查的 `(py312)` 环境运行：

```bash
cd /workspace
tar -xzf qwen3-reranker-baseline.tar.gz
cd qwen3-reranker-baseline
bash setup.sh
bash run_smoke.sh
```

小测试每个模型跑6题：单论文、双论文、三论文各2题。首次运行会从 Hugging Face 下载固定版本权重，下载等待不计入评分延迟。缓存默认放在 `/workspace/hf-cache`，请预留约30GB磁盘空间；显存与磁盘是两回事。

小测试通过后，运行全部593题：

```bash
bash run_full.sh
```

结果在 `results/comparison/report.md`，汇总数据在同目录 `summary.json`，逐题变化在 `per_question_changes.json`。后者的 `delta` 为重排减 Dense，正值改善，负值退化。每个模型所有50条候选的分数与顺序保存在 `results/full-06b/items/`、`results/full-4b/items/`，便于后续逐题追因。

中断后可以重跑同一命令，已完成的题会校验后跳过。参数、代码、依赖版本或数据改变时会拒绝复用目录；需要新建输出目录。每次重启都会重新预热。不要同时启动两份实验抢同一张GPU。

依赖装在包内 `.venv`，复用当前 `(py312)` 的 PyTorch/CUDA；不重装已验证的 PyTorch 2.13.0+cu132。先不要安装 FlashAttention、vLLM 或训练框架。如 Hugging Face 无法访问，先解决模型下载连通性；模型下载失败不代表显存不足。

## 固定的实验条件

| 项目 | 设置 |
|---|---|
| 模型 | Qwen/Qwen3-Reranker-0.6B；Qwen/Qwen3-Reranker-4B |
| 模型版本 | 0.6B：e61197ed45024b0ed8a2d74b80b4d909f1255473；4B：22e683669bc0f0bd69640a1354a6d0aebcfeede5 |
| 候选 | 原有 text-embedding-3-small Dense Top50，不再召回 |
| 模型输入 | 改写后的问题 + 每条候选的 heading_path 和正文 |
| 评分 | 官方 yes/no 模板，最后位置的 logit(yes) − logit(no) 降序 |
| 精度 | BF16，不量化 |
| 批大小 | 4个 query-document 对；同一道题的50条候选分批评分 |
| 最大长度 | 2048 tokens，含指令、问题、候选和评分后缀 |
| 注意力实现 | PyTorch SDPA |
| 延迟 | 预热后每题50候选的分词、GPU评分、排序；不含模型加载、embedding、答案生成、评审 |

模板按照 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 实现。原始 logit 差与官方 yes/no softmax 分数排序等价，同时避免将很大的分数压到1后产生假并列；并列时保留 Dense 顺序。这个分数不是“答案正确的置信度”。

`max_length=2048` 是本次实验设置，不是模型的上下文上限。每题记录原始 token 长度与截断数量；若发现截断，应先检查哪些题受影响，再用相同更大长度为两个模型另开实验，不能只给一个模型加长上下文。只输出最后一个位置的 logits 以节约显存；没有裁掉评分前文。

## 题目与评估规则

| 分组 | 数量 | 说明 |
|---|---:|---|
| single500 | 500 | 既有单论文改写题 |
| cross2 | 85 | 既有跨两篇论文题，经过全文测试筛选 |
| cross3 | 8 | 既有跨三篇论文题，经过全文测试筛选 |

模型只接收 `queries.jsonl` 的问题和 `documents.jsonl` 的候选内容。`labels.jsonl` 的证据标注只在排序完成后算指标，不传给模型。正文去重后共12687个块；数据文件和来源缓存的 SHA256 在 `data/manifest.json`。

指标包括 Hit@1/5/10/50、Recall@1/5/10/50、MRR@5/10、all_evidence@1/5/10/50，另记录 P50/P95、吞吐、GPU峰值已分配显存、截断数量。

- Hit@5：前5块至少命中一条标注证据。
- Recall@5：前5块覆盖的标注证据比例；单论文沿用历史“取最好的标注者证据组”口径。
- MRR@5：前5块中第一条标注证据排名的倒数，没有则0。
- all_evidence@5：Recall@5=1的题目比例。跨论文必须覆盖每篇论文的必要事实，才算找齐。
- Recall@50：候选池不变，因此重排前后应相同。候选池没有的内容，重排无法补回来。

跨论文沿用原实验的“论文内必要事实引文”匹配，导出时已核对与 supporting_chunk_ids 完全一致；没有使用更宽松的段落碎片匹配。它与单论文标注方法不同，因此分组汇报，不混成一个总分。

**这里的证据召回不等于此前 LLM 对最终答案计算的证据覆盖度。** 本次没有生成答案，也没有调用评审模型；不能从检索指标直接得到回答正确率或回答置信度。与 DeepSeek 的历史排序可以看质量趋势，但云端延迟和这台GPU的延迟不能当作同机测速。

## 微调前的边界

这些题已经用于开发、诊断和方案选择，93道跨论文题还经过“全文能答对”的筛选。它们适合当前模型对比，但不是新的盲测集。这轮不把它们用作训练样本。

后续做SFT时，先划分训练、验证、测试论文；一题涉及的多篇论文需要放在同一侧，避免跨论文泄漏。需额外准备未参与开发的测试题，不能在这593题上反复挑模型后把结果宣称为独立测试性能。非标注块也不能直接全部当负例，需审核困难负例。

## 检查状态

本地已验证数据完整性、593题 Dense 指标与历史结果一致，以及指标、截断和排序异常处理。2026-11-07已在租用服务器完成两个模型各6题小测试、各593题全量测试；结果已取回本地并从全部逐题排序重新核算。实测数据见 `Qwen3重排基准测试-2026-11-07.md` 和 `results/comparison/report.md`。

另外已下载两个固定版本的真实分词器：各29650对输入、最长484 tokens、2048长度下截断0对。首题50对的编码结果与官方示例逐 token 相同。记录在 `tokenizer-validation.json`。用随机小型 Qwen 网络在本地 CPU 核验过“只输出最后位置 logits”与完整 logits 的最后位置一致；这只是实现检查，不是两个真实模型的推理结果，也不能替代服务器 CUDA 验证。

离线检查可随时执行：

```bash
.venv/bin/python verify.py
.venv/bin/python -m unittest -v test_baseline.py
```
