# SC68：固定前五段资料，比较两版回答提示词

按相同 qid、问题、参考答案及两条检索路径前五段的编号、正文和顺序逐项核对。每轮分别对 Dense 与重排的匿名回答评分，下表按 qid 比较两轮的模型判定。

| 检索方案 | 旧版正确 | 新版正确 | 对→对 | 对→错 | 错→对 | 错→错 | 涉及待定 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 39 | 42 | 39 | 0 | 3 | 26 | 0 |
| rerank | 65 | 65 | 64 | 1 | 1 | 2 | 0 |

## 目标题、证据不足题与本轮回退

### 1805.09821-q01 · incorrect → correct

Which baselines evaluate cross-lingual document classification on the eight-language Reuters subset with balanced class priors?

参考答案：aggregation of multilingual word embeddings; multilingual sentence representations / we use MultiCCA word embeddings published by BIBREF3; joint multilingual sentence representations

新版回答：根据资料，该八语言 Reuters 子集（MLDoc）上的基线使用两类方法：基于多语言词嵌入的方法和基于多语言句子嵌入的方法，二者互为补充，并为全部语言迁移方向提供强基线结果[1][2][4]。MLDoc 被描述为覆盖 8 种语言的平衡 Reuters 子集，任务是 CCAT、ECAT、GCAT、MCAT 四类主题分类，按分类准确率评估[5]。

引用判定：supported。评审说明：B正确指出两类基线：基于多语言词嵌入和基于多语言句子嵌入的方法，并说明为全部语言迁移方向提供强基线；引用[1][2][4]原文支持这两类互补方向及全迁移方向结果。

旧版回答：根据现有资料无法回答这个问题。

拒答原因：资料[1][2][3][4]提到为八种语言的所有迁移方向提供了基于多语言词嵌入和句子嵌入的基线结果，但未说明这些基线是否在“类别先验均衡（balanced class priors）”的八语言 Reuters 子集上评估；资料[5]虽说明 MLDoc 是覆盖8种语言的平衡子集（balanced subset），但未提及具体哪些基线在其上评估。

### 1809.08935-q02 · incorrect → incorrect

What evaluation metric is used for the balikasg essay-level prediction system in the CAp 2018 challenge?

参考答案：Accuracy metric / accuracy / Accuracy

新版回答：根据现有资料无法回答这个问题。  
拒答原因：资料只提到组织者提出了一种使用混淆矩阵和先验知识、对远离真实类别的预测施加更高惩罚的成本度量（[1][2][3]），并说明该任务被建模为有序分类问题（[4]），但没有给出该评测指标的名称，也没有明确说明它就是 CAp 2018 challenge 中 balikasg 系统所采用的评测指标；资料[5]仅报告该系统在测试数据上的准确率为 98.2%，未说明这是否为官方评测指标。

引用判定：incomplete。评审说明：B 虽引用[5]提到测试数据准确率 98.2%，却未据此回答评测指标为 accuracy，仍拒答，属 incorrect；其引用[1][2][3][4]支持有序分类与成本度量，但未引用能直接支持 accuracy 作为评测指标的关键断言，引用不完整。

旧版回答：根据现有资料无法回答这个问题。  
拒答原因：资料[1]只提到组织者提出了一种使用混淆矩阵和先验知识的代价度量，但未给出该评价指标的名称；资料[2][3]只说明该度量对远离真实类别的预测施加更高惩罚、且上对角线误分类代价更高，仍无法确认具体指标名称。

### 1908.05925-q03 · incorrect → incorrect

Does CAiRE's WMT 2019 German-to-Czech unsupervised translation system evaluate its MUSE alignment independently of translation performance?

参考答案：No / No

新版回答：根据现有资料无法回答这个问题。
拒答原因：资料仅说明该系统使用 MUSE 对齐德语和捷克语的嵌入（[1][2][4][5]），但未提及是否对 MUSE 对齐本身进行独立于翻译性能的评测，也未报告任何单独的对齐质量指标。

引用判定：incomplete。评审说明：问题问该系统是否独立于翻译性能评估MUSE对齐，参考答案为No。A以资料不足为由拒答，但所引摘要和方法段只说明用MUSE对齐嵌入，未报告独立对齐评测，这正支持“没有独立评估”的否定回答；拒答按规则算incorrect。引用虽列出相关段落，但未直接支撑否定结论，故incomplete。

旧版回答：根据现有资料无法回答这个问题。  
拒答原因：资料只说明系统使用 MUSE 对齐德语和捷克语的 BPE/词级嵌入 [1][2][4][5]，但没有提到是否对 MUSE 对齐本身进行独立于翻译性能的评估。

### 1906.04287-q02 · correct → incorrect

Does the original evaluation of Dual-channel Word Embedding (DWE) for Chinese include downstream tasks beyond word similarity and word analogy?

参考答案：No / No / No

新版回答：根据现有资料无法回答这个问题。  
拒答原因：资料仅说明DWE在“word similarity”和“word analogy”两个代表性任务上进行了评测[1][2][3][4][5]，但没有提供关于是否还包含这两个任务之外的下游任务的任何信息；方法介绍或实验设置未提及其他下游任务，不足以证明其未包含其他任务。

引用判定：incomplete。评审说明：B同样拒答，未回答“不包含其他下游任务”，属incorrect；引用列出两任务及实验设置，但未明确断言无其他任务，故incomplete。

旧版回答：根据现有资料，DWE 的评估只包括 word similarity 和 word analogy 两类任务，没有提到其他下游任务。[2] 明确说“Through the evaluation on both word similarity and word analogy tasks”；[4] 也说“We use two test tasks to evaluate the performance of different models: one is word similarity, and the other is word analogy”；[1] 和 [3] 同样只提到“two representative tasks”。因此，资料显示原始评估未包括 word similarity 和 word analogy 之外的下游任务。

## 全部 68 题逐题对照

| qid | 旧版重排组 | 新版重排组 | 旧版引用 | 新版引用 |
|---|---|---|---|---|
| 1602.07563-q01 | correct | correct | supported | supported |
| 1604.02201-q01 | correct | correct | supported | supported |
| 1605.05156-q04 | correct | correct | supported | supported |
| 1605.05166-q02 | correct | correct | supported | supported |
| 1607.00167-q02 | correct | correct | supported | supported |
| 1607.03895-q01 | correct | correct | supported | supported |
| 1610.00479-q01 | correct | correct | supported | supported |
| 1611.01116-q01 | correct | correct | supported | supported |
| 1611.01884-q01 | correct | correct | supported | supported |
| 1611.06322-q01 | correct | correct | supported | supported |
| 1612.09535-q01 | correct | correct | supported | supported |
| 1702.02367-q03 | correct | correct | supported | supported |
| 1703.04357-q01 | correct | correct | supported | supported |
| 1705.02023-q02 | correct | correct | supported | supported |
| 1707.02892-q04 | correct | correct | supported | supported |
| 1708.01065-q05 | correct | correct | supported | supported |
| 1710.10380-q01 | correct | correct | supported | supported |
| 1801.03615-q01 | correct | correct | supported | supported |
| 1802.02614-q02 | correct | correct | supported | supported |
| 1802.03052-q02 | correct | correct | supported | supported |
| 1802.06053-q04 | correct | correct | supported | supported |
| 1802.09059-q03 | correct | correct | supported | supported |
| 1803.02914-q01 | correct | correct | supported | supported |
| 1803.04579-q01 | correct | correct | supported | supported |
| 1803.06745-q01 | correct | correct | supported | supported |
| 1804.01155-q04 | correct | correct | supported | supported |
| 1805.04833-q02 | correct | correct | supported | supported |
| 1805.09821-q01 | incorrect | correct | incomplete | supported |
| 1805.11598-q02 | correct | correct | supported | supported |
| 1805.11937-q03 | correct | correct | supported | supported |
| 1807.03674-q01 | correct | correct | supported | supported |
| 1809.08935-q02 | incorrect | incorrect | incomplete | incomplete |
| 1811.00127-q02 | correct | correct | supported | supported |
| 1812.05813-q02 | correct | correct | supported | supported |
| 1901.09501-q01 | correct | correct | supported | supported |
| 1902.04094-q01 | correct | correct | supported | supported |
| 1902.06734-q03 | correct | correct | supported | supported |
| 1903.10318-q03 | correct | correct | supported | supported |
| 1905.08067-q03 | correct | correct | supported | supported |
| 1906.01749-q04 | correct | correct | supported | supported |
| 1906.04287-q02 | correct | incorrect | supported | incomplete |
| 1908.05908-q01 | correct | correct | supported | supported |
| 1908.05925-q03 | incorrect | incorrect | incomplete | incomplete |
| 1908.11664-q01 | correct | correct | supported | supported |
| 1909.05372-q01 | correct | correct | supported | supported |
| 1909.09070-q01 | correct | correct | supported | supported |
| 1909.11706-q02 | correct | correct | supported | supported |
| 1909.12016-q03 | correct | correct | supported | supported |
| 1910.07154-q03 | correct | correct | supported | supported |
| 1910.07924-q02 | correct | correct | supported | supported |
| 1910.10670-q01 | correct | correct | supported | supported |
| 1910.13793-q02 | correct | correct | supported | supported |
| 1911.01248-q01 | correct | correct | supported | supported |
| 1911.03324-q02 | correct | correct | supported | supported |
| 1911.04474-q02 | correct | correct | supported | supported |
| 1911.07620-q01 | correct | correct | supported | supported |
| 1911.09247-q05 | correct | correct | supported | supported |
| 1912.10162-q03 | correct | correct | supported | supported |
| 1912.11637-q02 | correct | correct | supported | supported |
| 2001.04346-q03 | correct | correct | supported | supported |
| 2002.04326-q02 | correct | correct | supported | supported |
| 2002.04374-q02 | correct | correct | supported | supported |
| 2002.09616-q03 | correct | correct | supported | supported |
| 2003.03131-q01 | correct | correct | supported | supported |
| 2003.11531-q01 | correct | correct | supported | supported |
| 2003.12139-q02 | correct | correct | supported | supported |
| 2004.01820-q04 | correct | correct | supported | supported |
| 2004.02401-q02 | correct | correct | supported | supported |

[本轮逐题回答、来源原文和评审意见](review.html) · [上一轮报告](../sc68-refusal-v2/review.html)
