# 评测题集（从 Qasper test 改写 / 构建）

这些是 day1、day2 实验用的题集，从工作区 `/Users/bia/Documents/Codex/2026-11-02/wo/outputs/` 原样复制进项目（2026-11-07），内容没有改动。
来源数据：[Qasper](https://allenai.org/data/qasper)（Dasigi et al., 2021，CC BY 4.0）的 **test** 部分。

⚠️ 这些题都来自 Qasper **test**：可以当开发 / 对照用，**不能当训练数据**（做 SFT 时只能用 Qasper train）。最终结果建议在没用过的 validation 部分上另外确认。

| 目录 | 题数 | 说明 |
|---|---|---|
| `qasper/pilot68/` | 68（另有暂缓 32） | 从 100 篇论文各抽一题（seed=42）改写成"不用先指定论文也能看懂"的问题；`questions_ready_68.jsonl` 是正式题，`queries_only_68.jsonl` 只有问题，`review_100.jsonl` 是逐题审阅记录。改写时看过证据和答案，**有泄漏风险**，见 day1 第 5 节 |
| `qasper/sc500/` | 500 | 另外 270 篇论文（排除了 pilot 的 100 篇），seed=20261103。改写只看标题、摘要和原题；`review.jsonl` 是 788 道候选的审核记录 |
| `qasper/cross16/` | 16 | 第一批双论文题（32 篇来源论文） |
| `qasper/cross100/` | 100（90 道双论文 + 10 道三论文） | 185 篇来源论文，和 cross16 不重合；`fulltext-screen/` 里是"指定全文能答对"筛选后的 93 题（85 + 8）以及排除的 7 题 |

每个目录的 `manifest.json` 记录了构建方法、随机种子和文件哈希。部分 manifest 里写的是本机绝对路径，换电脑时需要对应调整。
