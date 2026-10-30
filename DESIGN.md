# RAG 知识库 —— 设计文档

> 最后更新：2026-10-28
> 状态标记：✅ 已实现 ・ 📌 已决定（未实现） ・ 🚧 设计中 ・ ❓ 待定

---

## 1. 项目定位

- **起点**：个人笔记知识库，包括 Markdown、纯文本、Word、PDF 和图片。笔记主要来自飞书导出。
- **目标**：做成可以写进简历的项目，并按企业级项目的要求来设计，比如权限、增量更新和评测。
- **模型**：生成用 DeepSeek（云端 API）；embedding 倾向于本地部署（见第 5 节）。
- **向量库**：Milvus。
- **原则**：先手写，把每一环搞懂，再按需要引入框架。

## 2. 总体流程

```
建库：文件 ─① 解析→ Document ─② 切分→ Chunk ─③ 向量化→ 向量 ─④ 入库→ Milvus
问答：问题 → 向量 → 在 Milvus 检索相近的 chunk → 和问题一起交给 LLM → 带出处的回答
```

当前进度：① 解析 ✅，② 切分 ✅（MVP 版），③ ④ 和问答部分还没开始。

## 2.1 MVP 计划（2026-10-28 决定）

目标：先做出一个能**从头跑到尾**的最小版本，再回头补功能。

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 切 chunk + 离线测试 + CI | ✅ |
| M2 | 向量化 + 存入 Milvus Lite + 检索小工具 | ✅（PR #1，Mac 和 CI 的 Linux 上都验证过） |
| M3 | 问答接口（FastAPI）+ 标出处 | ✅（PR #2，见第 8 节） |
| M4 | Docker 镜像 + CD：打 tag 时自动构建镜像，推送到 GitHub 的镜像仓库 | ✅ Dockerfile（PR #3）+ CD（PR #5，`v0.1.0` 已发布，见第 9 节） |

🎉 **2026-10-30：MVP 完成。** 推送 `v0.1.0` → GitHub Actions 自动测试、构建 amd64 + arm64 镜像 → 推送到 `ghcr.io/sty-wq/rag` → Mac 上 `docker pull` 后直接运行，问答正常。

**已经决定的**

- 代码放到 **GitHub 公开仓库**（简历用）。公开之前要检查：不提交密钥、私人笔记和不相关的文件
- 提问方式：**HTTP API（FastAPI）**；建库先用命令行
- 向量库：**Milvus Lite**。它就是本地一个文件，不用装服务器，和正式版 Milvus 用的是同一套接口（`MilvusClient("./xxx.db")`），以后换成服务器版只改连接地址。CI 里也能直接跑
- CD：**Docker 镜像**
- embedding：**本地部署**，用 Ollama 跑 `qwen3-embedding:0.6b`（见第 5 节）
- **git 工作流**：每个里程碑开一个分支（比如 `feat/embedding`），推到 GitHub 后开 Pull Request，CI 通过了再合并进 main。main 始终是能跑、测试全绿的版本

**MVP 先不做**（停车场）：增量更新（MVP 每次全量重建）、疑似标题、第 ⑤ 步 LLM 切分、摘要索引、评测集、权限、MySQL、飞书接入、老 `.doc` 格式、README

**停车场：部署方案**（2026-10-28 新增；2026-10-30 **暂停**，先只在 Mac 上验证，以后换一台带宽更大的服务器再继续）

倾向的方案：**服务全部跑在家里的 Jetson 上，云服务器只负责把外面的请求转进来**

```
外面的用户 ──HTTPS──▶ 云服务器（2 核 2G，有公网 IP）──隧道──▶ 家里的 Jetson（FastAPI + Ollama + Milvus Lite）
```

- 好处：干活的部分（embedding、检索）都在 Jetson 上，能用 GPU；云服务器只做转发，2G 内存足够。"服务器上必须用和建库时同一个 embedding 模型"的问题也就不存在了
- 内网穿透的几种做法：frp（云服务器跑 frps，Jetson 跑 frpc）、SSH 反向隧道（`ssh -R` 配合 autossh）、WireGuard / Tailscale 组网 + 云服务器上用 Nginx 反向代理。待选
- 要验证的：Jetson 是 ARM64 架构，Ollama 和 Milvus Lite 在上面能不能正常跑；Docker 镜像要构建 ARM64 版本（CD 里用 `docker buildx` 同时构建 amd64 和 arm64，正好是一个 CD 的学习点）
- ⚠️ **安全**：接口暴露到公网后，`/ask` 每次都会调用 DeepSeek，谁都能刷就等于谁都能花你的钱。上线前至少要做到：**API Key 鉴权**、**HTTPS**、**限流**

**Jetson 已经做好的准备**（2026-10-30）
- 环境：aarch64，JetPack 6（L4T R36.4.4），Docker 28.3.3；Ollama 还没装
- 新加了一块 128G 的 NVMe 硬盘，格式化成 ext4，挂载在 `/data`。`/etc/fstab` 里用 **UUID + `nofail`**：设备名会变，UUID 不会；盘不在时也不会卡住开机
- 修复了 fstab 里一条失效的旧配置（指向一块已经不在的盘，没加 `nofail`，重启时可能进入紧急模式）
- Docker 的数据目录改到 `/data/docker`（`/etc/docker/daemon.json` 的 `data-root`，NVIDIA 的配置保留）；并加了 systemd 配置 `RequiresMountsFor=/data`：**`/data` 没挂上时 Docker 不启动**。教训：之前的数据盘丢了以后，Docker 悄悄把数据写到了系统盘上
- 镜像不用从网上拉：`docker save | gzip` 导出成文件，拷过去再 `docker load`。Mac 和 Jetson 都是 arm64，镜像可以直接用
- 遇到的问题：
  - Jetson 上的 Docker **连不上 Docker Hub**（Docker 后台程序不走 Clash 代理）。以后从镜像仓库拉镜像时，要给 Docker 配置代理
  - 云服务器带宽小，通过它传几百 MB 的镜像很慢 → 这就是暂停的原因
  - 在 Linux 上，容器里默认**没有** `host.docker.internal`，Ollama 默认也只监听 `127.0.0.1`。到时候用 `--network host` 运行容器
- 安全加固（SSH 只允许密钥登录、关掉不用的 Jupyter 和远程桌面等）：这次是实验，暂时没做，正式部署前要补上

## 2.2 工程化：测试与 CI（✅ M1）

- **测试全部离线**：不调用 DeepSeek 和 OCR 服务商，不需要任何 key，也不花钱。需要大模型的地方用假客户端（`tests/conftest.py` 里的 `FakeLLM`），需要发 HTTP 请求的地方用 monkeypatch 替换 `httpx.post`
- **测试不依赖 `sample_notes/`**：测试数据都在测试代码里现造，所以私人笔记不提交到公开仓库也不影响 CI
- 代码检查：ruff，MVP 只开 `E9`（语法错误）和 `F`（pyflakes：未定义的名字、没用到的 import 等）。`cable_ocr/` 是单独集成进来的 OCR 项目，不在检查范围内
- CI：GitHub Actions（`.github/workflows/ci.yml`），每次推送到 main 和每个 Pull Request 都会自动运行：安装依赖 → ruff → pytest
- 开发依赖在 `requirements-dev.txt`，pytest 和 ruff 的配置在 `pyproject.toml`
- 已有的 OCR 集成测试（`tests/test_ocr_integration.py`）里有两个测试还是按旧代码写的（绝对路径、`ocr_result` 字段、全局列表），已经按现在的设计更新
- ✅ 2026-10-28：推送到 GitHub 公开仓库 [sty-wq/RAG](https://github.com/sty-wq/RAG)，CI 第一次运行就全部通过（22 秒）
- M2 新增的测试：向量化客户端用假的 HTTP 请求；**向量库用真的 Milvus Lite**（`tests/test_vector_store.py`，没装 pymilvus 的环境会自动跳过，CI 里一定会跑）；建库和检索的整体流程用假的 embedding 和假的向量库

---

## 3. 阶段一：解析（✅ 已实现，代码在 `main.py`）

### 3.1 Document 数据结构

| 字段 | 作用 |
|---|---|
| `id` | 文档的唯一标识。chunk 靠它知道自己属于哪个文档，增量更新时也靠它找到旧数据 |
| `name` | 文件名，用于展示 |
| `content` | 解析后的正文，**用 Markdown 表达结构** |
| `source_path` | 来源路径，回答时用来标注出处 |
| `meta_data` | 和格式相关的附加信息，**保持小而扁平** |

### 3.2 解析器注册表

- 用字典登记"后缀 → 解析函数"，相当于 Java 的 `Map<String, Function<Path, Document>>`。新增一种格式只需要写一个函数，再加一行登记。
- **所有解析函数必须返回同一种类型（Document）**。Python 不会检查这一点，只能靠约定守住。

| 格式 | 实现方式 | 状态 |
|---|---|---|
| `.md` / `.markdown` | 原样读取，保留 `#` 等结构 | ✅ |
| `.txt` | 先按 UTF-8 解码，失败再用 GB18030 | ✅ |
| `.docx` | python-docx：`Heading N` 样式转成 N 个 `#`，`HTML Preformatted` 样式转成 ``` 代码块 | ✅ |
| `.pdf` / `.jpg` / `.png` | 接入已有的 OCR 模块：有文字层就直接提取，没有才调用云端 OCR（默认百度） | ✅ |
| `.doc`（老格式） | Mac 上用 `textutil` 转成 docx，服务器上用 LibreOffice | ❓ |

### 3.3 处理流程与错误处理

```
遍历目录 → 按后缀找解析器 → 解析 → clean_text → 判断是否为空 → 放进对应的列表
```

- **异常分层处理**：解析函数只管抛出异常；遍历目录的循环统一 `catch Exception`，把出错的文件记进失败列表，然后继续处理下一个。这个循环不需要知道每个解析器会抛哪种异常，所以新增解析器时不用改它。
- **能就地处理的就地处理**：比如编码回退，只捕获 `UnicodeDecodeError`。其他异常交给上一层去记录。
- **结果分三类**：成功、失败（不支持的格式或解析出错）、内容为空。三类都会提示给用户。
- **`clean_text` 放在所有 Document 都会经过的地方**（`load_directory`），新增格式时会自动生效，不会漏掉。
  - 统一换行符（先替换 `\r\n`，再替换 `\r`，**顺序不能反**），并去掉首尾空白。
  - **保留 `\n` 和 `\t`**：它们是结构信息，切分阶段要用。删掉就找不回来了。
- **空内容的判断放在清洗之后**：一个只有空白字符的文件，要清洗之后才会变成空字符串。

### 3.4 关键设计决策

| 决策 | 考虑过的备选 | 理由 |
|---|---|---|
| 各种格式都**统一转成 Markdown** 来表达结构 | 统一转成 PDF | PDF 只记录"字画在哪"，标题和代码块的信息会丢失；Markdown 能保留结构，下游只需要认一种结构语言 |
| docx **直接解析** | docx 转成 PDF 后再提取 | 直接解析能拿到样式里的标题信息，而且不依赖 Word 或 LibreOffice；转 PDF 只作为兜底，或者用在图片很多的文档上 |
| PDF **先提取文字层，再考虑 OCR** | 一律走 OCR | 文字层又快又准，而且不花钱 |
| `meta_data` 只放小字段，OCR 的完整结果只记一个路径 | 把整个 OCR 结果放进去 | metadata 会跟着每个 chunk 一起存进 Milvus，放大对象会被复制很多遍 |
| doc id 用**路径**的 MD5 | 用内容的 MD5 | id 要回答的是"这是哪个文件"，文件改了 id 不能变，这样才能替换掉旧数据。内容哈希留着判断"文件有没有改过" |

### 3.5 已知问题与待办

- [ ] 三个解析函数里组装 Document 的代码重复（id、name、source_path）
- [ ] `load_directory` 返回 3 个值，以后可以改成一个结果对象，或者合并成一个 `skipped` 字典
- [ ] Markdown 开头的 front matter 应该解析进 meta_data，而不是留在正文里
- [ ] 以后实现 `.doc` 老格式的支持
- [ ] 去掉目录页、页眉、页脚、页码等噪音（在各个解析器里做，只删有把握的部分）
- [ ] 手写图的 OCR 有错字，方框和箭头的结构也丢了：**识别质量会直接影响检索效果**
- [ ] 示例 docx 里的代码在源文件中就丢了换行（XML 里没有任何换行标记），**解析阶段无法恢复**
- [ ] 每次运行都会生成一个新的 OCR 任务目录，需要确认缓存命中，不会重复扣费
- [ ] doc id 目前用的是调用时传入的路径，会受调用方式影响（见第 6 节）

---

## 4. 阶段二：切分（🚧 进行中）

### 4.1 设计原则

1. **结构是给程序用的，LLM 不太需要。** 先用规则尽量把结构挖出来，**能用规则做的就不花 token**。
2. **LLM 只出方案，不改写原文。** LLM 输出的是编号、层级和摘要，原文由程序来拼，一个字都不会变，出处才可信。
3. **结构是候选，不是事实。** 文档结构可能**标错**（比如示例 docx 的题目 4 把"参考代码"标成了同级标题），也可能**没标**（比如用加粗代替标题）。所以要先验证，再使用。
4. **方案必须能被程序检查。** 不合格就重试，还不合格就退回规则切分。
5. **结果要稳定。** 用内容哈希缓存切分方案：内容没变，就复用上次的方案，不再调用 LLM。

### 4.2 切分流程

```
② 规则：提取候选结构（明确的标题 + 疑似的标题），切成 Section；4B 小模型为每节写摘要，生成大纲
②'LLM：只看大纲，修正标题层级、确认疑似标题        ← 很便宜，每篇都做
③ 规则：按修正后的结构切成章节，做长度检查；没超限的直接成为 chunk
④ 规则：超长的章节按段落切，再把相邻段落合并到不超限
⑤ 规则切不动的：交给 LLM 出切分方案（第一版先不加小模型，看有多少章节会走到这一步再决定）
```

**大纲示例**（只发给 LLM 这一份，不发全文）：

```
[3] 题目4      约 300 字    开头：有一天，小红面临了一个新的数学难题……
[3] 参考代码    约 1500 字   开头：```cpp
```

→ LLM 输出：参考代码应该是题目4 的下一级。

**成本估算**：一份大纲大约 100–200 token，1000 篇文档总共约 20 万 token，按 DeepSeek 当前高峰价格（输入 $0.3 / 百万 token）计算不到 $0.1。

### 4.3 Section 与大纲校验（📌 已决定）

```
Section（章节）
  id            文档内的顺序编号（根节点为 0，标题依次为 1, 2, 3…），LLM 用它来引用
  doc_id        属于哪个文档
  title         标题文字（根节点为空）
  level         几级标题（根节点为 0）← 唯一说了算的层级信息
  parent_id     上级 section 的 id（由程序根据 level 计算，根节点为 None）
  content       本节自己的原文，不含子节
  is_candidate  是否是规则猜出来的疑似标题
  summary       摘要（None 表示还没生成，或者不需要生成）

SectionFix（LLM 只输出需要改的那几条）
  section_id    改哪一节
  new_level     改成几级；不是标题就写 None
  reason        一句话说明原因
```

**设计要点**

- **每篇文档都有一个 level 0 的根节点**：第一个标题之前的内容放在这里；没有标题的文档只有根节点；顶层章节的上级就是根节点。
- **content 不包含子节的内容**：每段原文只属于一个 Section，不会重复。
- **只有 level 说了算**：原文顺序不会变，所以只要确定每个标题是几级，整棵树就唯一确定了。LLM 只改 level，`parent_id`、同级序号、"2.1.1" 这样的编号、字数都由程序计算。这样就不会出现多个字段互相矛盾的情况。**计算 parent_id 要写成独立的函数**，因为应用 LLM 的修正之后还要再算一次。
- **LLM 只列出要改的章节**：输出更短，不需要"未修改"这种特殊值；用 `section_id` 引用章节，不复制整个 Section。`reason` 方便检查和调试。

**摘要**

- 由大约 4B 的小模型生成，只回答"这段讲的是什么"，不负责切分。需要切分时，由大模型给出切分点。❓ 用哪个模型、放在云端还是 Mac 本地跑，选型时再定
- ✅ **目前暂时用 DeepSeek**（`DEEPSEEK_MODEL`，读取 `.env.ocr`），并**关闭思考模式**：V4 默认开启思考模式，总结这种简单任务用不上，关掉更快也更省。调用部分封装在 `llm.py`，以后换模型只改这一处
- ✅ 实现：`outline.py`（摘要和大纲）、`llm.py`（DeepSeek 客户端：可重试的错误自动重试，401/400 这类错误直接报错；累计 token 用量）
- 暂定参数：`SHORT_CONTENT = 100`（不到 100 字不总结）、`MAX_SUMMARY_INPUT = 8000`（超长章节只取前 8000 字去总结）、`temperature = 0`
- 某一节总结失败不会影响其他节，大纲里那一节退回显示"开头一句"
- 不生成摘要的情况：
  - content 为空（只有标题的父章节）：大纲里内容留空，LLM 能看到它的标题和子节的摘要
  - content 很短（阈值待定，暂定 100 字以内）：直接把原文放进大纲
- 摘要以后可以直接复用为摘要索引（见 4.7）
- ✅ 用内容哈希缓存，没改过的章节不重新总结。缓存的 key 是"提示词版本 + 模型 + 标题 + 内容"的哈希：改了提示词或换了模型，旧缓存会自动失效。缓存文件是 `.rag_cache/summaries.json`（不提交到 git）
- 成本：章节数远多于文档数，这是整个流程里调用模型次数最多的一步

#### 4.3.1 第 ②' 步的实现（✅ `structure_fix.py`）

```
Section 列表 → 生成大纲 → 有缓存就用缓存，没有才问 LLM（JSON 输出模式）
           → 程序检查每一条修正 → 应用修正 → 重新计算 parent_id → 检查原文完整
```

- **提示词里的示例故意不用测试文档的真实情况**：如果提示词里直接写了"参考代码应该归到题目下面"，测试通过也说明不了模型真的会判断，这就像考试前把答案泄露了
- **JSON 输出模式**：请求里加上 `response_format: {"type": "json_object"}`，提示词里必须出现 "json" 这个词，并给出格式示例。DeepSeek 文档说这个模式偶尔会返回空内容，所以遇到空内容会重试；返回的不是合法 JSON 也会再试一次
- **程序检查**（不合法的条目丢弃并说明原因，合法的照常采用）：section_id 必须存在、不能是根节点、不能重复；new_level 必须是 1–6 或 null；级别没有变化的忽略。LLM 把数字写成字符串（`"5"`）时宽容地转换一下
- **应用修正**：不修改原来的列表，生成一份新列表，方便对比修正前后
  - 改级别：只改 level，然后重新调用 `assign_parents()`
  - 不是标题：标题这一行**原样**并回上一节（明确的标题保留 `#` 标记，疑似标题就是原文那一行），上一节的旧摘要作废
- **退路**：LLM 调用失败时保持原结构，流程照样往下走；失败的结果不写缓存
- **缓存**：key 是"提示词版本 + 模型 + 是否思考 + 大纲"的哈希，文件是 `.rag_cache/structure_fixes.json`
- `check_lossless()` 比较时会去掉每行开头的 `#`：标题是几级属于"结构"，不属于"文字"。改了级别，文字没变，检查照样通过
- ✅ **开启思考模式**（`USE_THINKING = True`）。2026-10-28 用 3 篇样例文档实测，提示词完全相同：

  | | 关闭思考 | 开启思考 |
  |---|---|---|
  | docx：参考代码 → 题目4 的下一级 | ❌ 漏改 | ✅ 改对 |
  | 两个 md：结构本来就是对的 | ✅ 没乱改 | ✅ 没乱改 |
  | 输出 token（3 篇合计） | 18 | 622 |

  - 结论：提示词没问题，**漏改的原因是模型没有思考的空间**。输出 token 多了约 35 倍，但绝对成本仍然很低（1000 篇文档估计不到 $0.3）
  - 从思考过程能看出：它是靠**摘要**把"参考代码"和"题目4"对上的。这验证了"大纲里放摘要，而不是只放开头一句"的设计
  - 思考模式下 JSON 输出模式可用（文档没写，这次实测确认）
  - 开了思考模式以后，运行结果会打印 LLM 的思考过程，方便调试
  - ⚠️ 目前只有 1 个有错误的样例，结论还不稳。要下定论，需要评测集（见 4.8）

### 4.4 硬性规则（任何切分方式都要遵守）

- 代码块、表格、列表不能拆开
- 每个 chunk 不超过 N 字（N 待定，需要通过评测来确定）
  - N 应该**远小于** embedding 模型的输入上限（常见做法是几百 token）。一段文字只会变成一个向量，内容越多，细节越容易被"平均"掉，检索就越不准
  - 程序要自己控制长度：超出上限时，有的平台会报错，有的会**悄悄截断**
- **代码块不能拆，但它可能比 embedding 模型的上限还长**：这种情况下必须用摘要索引（见 4.7），拿摘要去向量化，命中后返回完整代码
- 每个编号恰好出现一次，并保持原来的顺序
- 代码块里以 `#` 开头的行（比如 `#include`）不是标题

### 4.5 切分方式（LLM 提示词的判断顺序，逐段判断，同一篇文档可以混用）

1. 有标题结构 → 按章节切
2. 有问答结构 → 一问一答作为一个 chunk。问答有哪些写法：❓ 待整理，包括 `问：/答：`、`Q:/A:`、"题目 / 解析 / 参考代码" 等
3. 有语义边界 → 把同一话题的相邻段落合在一起
4. 都没有 → 标记为"按长度切"，交给程序处理

每个 chunk 要写明它用的是哪种切分方式，方便以后评测时按方式统计效果。

### 4.6 Chunk（✅ MVP 版，`chunks.py`）

```
Chunk
  id            md5(文档id : 章节id : 本节第几块)，同样的结构每次算出来都一样
  doc_id        属于哪个文档
  doc_name      文件名（出处）
  source_path   来源路径（出处）
  section_id    来自哪个章节
  heading_path  标题路径，例如 "柠檬微趣CPP.docx > 题目4 > 参考代码"
  order         在整篇文档里是第几块
  text          原文：检索命中后交给 LLM，也展示给用户
  embed_text    拿去向量化的文字："标题路径 + 原文"，最多 EMBED_MAX_CHARS 字
```

- **出处信息冗余存在每个 chunk 上**：Milvus 没有 JOIN，回答时需要的信息要直接带在 chunk 上
- **标题路径从文件名开始**：只看到 `return result;}` 这样一小段代码时，也知道它属于哪个文件、哪道题。内容为空的父章节不产生 chunk，但会出现在子节的标题路径里
- **切分规则（MVP 版，覆盖设计里的 ③④）**：
  - 章节不超过 `MAX_CHUNK_CHARS`：整节就是一块
  - 超长的章节：先切成"段落"（按空行）和"代码块"，再把相邻的合并到不超过上限
  - **代码块不拆**，超长也单独成一块（向量化时截断，以后用摘要索引处理）
  - 超长的普通段落按句子切（。！？；等），单句还超长就硬切
- 暂定参数：`MAX_CHUNK_CHARS = 800`、`EMBED_MAX_CHARS = 4000`，以后用评测数据来定
- `check_chunks_lossless()`：检查每一节切成 chunk 之后文字一个字都没丢（忽略空白）
- 实测（docx，未做结构修正）：5 个章节 → 11 块；3 段超长代码各自成块

### 4.7 后续增强（📌 第一版之后，用评测数据决定要不要做）

- **摘要索引**（Multi-Vector）：用 LLM 写的摘要去检索，命中后返回原文。适合很长的代码块。**对超过 embedding 上限的代码块来说，这一项是必须做的。**
- **Contextual Retrieval**：让 LLM 给每个 chunk 写一小段背景说明，加在 chunk 前面再去向量化。
- 简历上的表述目标：类似"加入 X 之后，召回率从 A 提升到 B"这样有数据支撑的结论。

### 4.8 当前任务

- ✅ 第 ② 步：把 Markdown 切成 Section 列表、计算 parent_id（`chunker.py`）；写摘要、生成大纲（`outline.py`）
- ✅ 第 ②' 步：LLM 看大纲，输出 SectionFix 修正层级（`structure_fix.py`），细节见 4.3.1
- ✅ 真实 API 实测：开启思考模式后，"参考代码"被正确放到题目4 下面，md 文件没有被乱改
- ✅ 第 ③④ 步 MVP 版：章节 → chunk（`chunks.py`），见 4.6
- 📌 停车场（以后做）：
  - 疑似标题（`is_candidate`）的识别规则
  - **结构校验的评测集**：手写大约 10 份带标准答案的大纲（一半结构正确，一半各带一种错误），对比不同方案的发现率和误改率，避免只对着一个样例调提示词（过拟合）

---

## 5. 阶段三、四：向量化与入库（✅ M2，`embedder.py`、`vector_store.py`、`ingest.py`、`search.py`）

**用法**

```bash
python ingest.py sample_notes            # 建库：完整流程（摘要和结构修正会调用 DeepSeek，有缓存）
python ingest.py sample_notes --no-llm   # 建库：不调用 DeepSeek，只用规则切分
python search.py "你的问题" -k 5          # 检索：看看找出来的是哪几块、相似度多少
```

**配置都用环境变量**（代码里有默认值，换平台或放进 Docker 时只改环境变量，不改代码）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | 本地 Ollama；换成百炼、硅基流动的地址就是云端 |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | |
| `EMBED_API_KEY` | 空 | Ollama 不需要 |
| `EMBED_BATCH_SIZE` | `10` | 一次请求最多送多少条（百炼的上限是 10） |
| `MILVUS_URI` | `.rag_data/milvus.db` | 以 `.db` 结尾就是 Milvus Lite；换成 `http://地址:19530` 就是 Milvus 服务器 |

**向量库（Milvus）的设计**

- 集合名 `notes`，字段：`id`（主键，就是 chunk id）、`vector`、`doc_id`、`doc_name`、`source_path`、`section_id`、`heading_path`、`chunk_order`、`text`
- 索引：`AUTOINDEX` + **余弦相似度**（COSINE）：只看两个向量的方向是否接近。检索结果里的分数越接近 1 越相似
- **MVP 每次全量重建**：先删掉旧集合，再重新建表、插入
- 文字字段有长度上限（按 UTF-8 字节算，一个汉字 3 个字节，`text` 最多 65535 字节）。超过就截断，并在建库报告里提示。截断时不会把一个汉字切成两半
- **记下用哪个模型建的库**：建库时写一个说明文件（`milvus.db.meta.json`，记录模型、维度、块数、时间）。检索时如果发现当前配置的模型和建库时不一样，直接拒绝，提示重新建库
- Milvus Lite 同一时间只允许一个进程打开同一个 `.db` 文件，所以代码里用 `with VectorStore() as store:` 保证用完就关闭
- `run_ingest()` / `run_search()` 把 embedding、向量库、LLM 都作为参数传进去，测试时可以换成假的

**以后可以改进的**（📌 停车场，用评测数据决定）
- Qwen3-Embedding 官方建议**给"问题"加一句任务说明**再做向量化（文档不用加），据说能提升检索效果。MVP 先不加，以后用评测集对比效果

**为什么选本地 Ollama**

- 模型：`qwen3-embedding:0.6b`（约 640MB，1024 维，上下文 32K，支持 100 多种语言）。效果不够再换 `4b`（约 2.5GB，2560 维）。和阿里云百炼的 text-embedding-v4 是同一个系列
- 通过 Ollama 的 **OpenAI 兼容接口** `POST /v1/embeddings` 调用。百炼、硅基流动也是同样的接口，所以代码只写一份，以后在本地和云端之间切换只需要改配置（地址、key、模型名）
- Ollama 直接装在 Mac 上，能用上 M 系列芯片的 GPU；服务放在 Docker 里，通过 `host.docker.internal:11434` 访问 Ollama。镜像里不用装 PyTorch 和模型，体积小
- ⚠️ **换 embedding 模型，所有向量都要重新生成**：不同模型生成的向量不能混在一起比较。所以每个向量库都要记下它是用哪个模型生成的
- CI 里不跑 Ollama，测试用假的 embedding
- 安装时踩过的坑：`ollama pull` 报错 "redirect target not allowed ... resolves to non-public 198.18.x.x"。原因是代理软件的 fake-ip 模式返回了假地址，而新版 Ollama 为了修复安全漏洞（CVE-2026-85180），会拒绝非公网地址。解决办法：在代理的 fake-ip 例外列表里加上 `+.r2.cloudflarestorage.com`，或者临时关掉 fake-ip

候选的云端 embedding 模型（中文效果都不错，2026-10 的规格）：

| 模型 | 平台 | 单条输入上限 |
|---|---|---|
| text-embedding-v4（Qwen3-Embedding 系列） | 阿里云百炼 | 8,192 token |
| BAAI/bge-m3 | 硅基流动 | 8,192 token |
| Qwen3-Embedding-0.6B / 4B / 8B | 硅基流动 | 32,768 token |
| BAAI/bge-large-zh-v1.5 | 硅基流动 | 512 token |

- 选型时要比较：中文效果、价格、向量维度（决定 Milvus 的存储量）、输入上限
- 一个汉字大约是 0.6～1 个 token，具体因模型而异。准确数字以 API 返回的 `usage` 字段为准

---

## 6. 数据同步与文档身份

| 场景 | 谁是"真相" | 怎么同步 |
|---|---|---|
| 个人笔记 | 笔记文件本身，可以用任何编辑器修改 | **定期扫描对比**（类似 make）：用路径 id 找到旧记录，再用内容哈希判断有没有改过 |
| 企业项目 | 系统本身就是文档平台 | **控制修改入口**：文档通过上传接口进入系统，由系统分配 id，修改和删除也都走接口 |

- 向量库只是笔记的"索引"，**RAG 系统只读不写原始文件**。索引坏了可以随时删掉重建。
- 当前的路径 id 只是个人笔记阶段的临时方案。**id 的生成逻辑始终只放在 `md5_calculate_doc_id` 这一个函数里**，以后换方案只需要改这一处。
- 路径 id 的已知缺点：文件改名或挪位置，会被当成"删掉一个旧文件、新增一个新文件"，旧数据需要另外清理。

---

## 7. 企业化需要考虑的事（❓ 第一版之后）

- **权限**：每个 chunk 都带上"谁能看"的信息，检索时按当前用户过滤。这会影响 Milvus 的字段设计。
- **评测**：检索的召回率、回答是否忠实于原文。这是简历上最有说服力的部分。
- **存储分工**：Milvus 存向量、id 和用于过滤的字段；MySQL 存正文、权限和版本。两边要同步更新。
- 等四个阶段都打通后，用标准流程（需求分析 → 架构设计）把企业版的需求正式梳理一遍。

---

## 8. 问答与接口（✅ M3，`answer.py`、`api.py`）

**流程**

```
问题 → 检索前 k 块 → 丢掉相似度低于 MIN_SCORE 的 → 一块都不剩：直接回答"无法回答"（不调用 LLM）
                                               → 还有剩下的：编号后交给 DeepSeek → 从回答里找出 [1] [2] → 标出处
```

- **只根据资料回答**：提示词要求只用资料里的信息，用 `[编号]` 标出处；资料里没有答案时，必须原样回答"根据现有资料无法回答这个问题。"
- **相似度门槛**：`MIN_SCORE = 0.35`（📌 暂定，可以用环境变量 `RAG_MIN_SCORE` 调整）。依据只有一个样例：相关的块 0.44～0.60，不相关的 0.34～0.38。门槛只负责去掉明显不相关的，剩下的交给 LLM 判断
- 之所以要门槛：检索"取前 k 个"**永远会凑满 k 个**，哪怕后面几个和问题毫无关系
- 一块都不剩时不调用 LLM：既省钱，也避免它"硬编"一个答案
- 回答用非思考模式（快），最多 1500 个 token
- 返回的每个出处都带 `cited`：回答里有没有真的引用它

**接口**

```bash
.venv/bin/uvicorn api:app --reload      # 启动；然后打开 http://127.0.0.1:8000/docs 在网页上试
```

| 接口 | 说明 |
|---|---|
| `GET /health` | 服务是否正常，向量库是用哪个模型建的 |
| `POST /ask` | 请求 `{"question": "…", "k": 5}`，返回 `{question, answer, found, sources[]}` |

- 错误码：**422** 请求不合法（问题为空、k 不在 1～20 之间）；**503** 还没建库，或者 embedding 模型对不上；**502** 调用 DeepSeek 或 embedding 失败
- 请求和返回都用 Pydantic 模型定义，`/docs` 页面会自动生成接口说明
- 依赖（embedding、向量库、LLM）在服务**启动时**创建、**关闭时**释放（FastAPI 的 lifespan）；`create_app(deps)` 可以传入假的依赖，测试时用
- ⚠️ Milvus Lite 同一时间只允许一个进程打开数据库文件：**服务运行时不能跑 `ingest.py`**，要重新建库，先停掉服务

**测试**：`tests/test_answer.py`（问答逻辑，全部用假的依赖）；`tests/test_api.py`（用 FastAPI 的 TestClient 发请求；没装 fastapi 的环境会自动跳过，CI 里一定会跑）

**以后可以改进的**（📌 停车场）
- 用评测集来定 `MIN_SCORE`
- "有没有找到答案"现在靠精确匹配那句固定的话来判断，比较脆弱
- 检查回答里的每句话是不是真的有资料支持（忠实度评测）
- 流式输出（边生成边显示）、多轮对话
- 复杂问题用思考模式
- ✅（已加进提示词）**资料自相矛盾时要指出来**：实测"红黑树"的问题时，原文的"解析"说从前往后遍历、删除元素，"代码"却是从后往前遍历、插入元素。回答把两种说法拼在了一起，没有指出矛盾。**RAG 会忠实地照搬资料里的错误**，回答质量的上限就是资料的质量。可以在提示词里要求"资料之间有矛盾时明确指出"

**实测**（2026-10-28，Mac）：
- "红黑树怎么找比当前值大的最小值" → 回答用到了 3 块资料，每句都标了出处；第 4 块（题目4，相似度 0.384）过了门槛，但 LLM 没有引用它（`cited: false`）
- "今天天气怎么样" → 所有块都低于 0.35，没有调用 LLM，直接回答"无法回答"（`sources: []`）

---

## 9. Docker 与 CD（✅ M4）

**Dockerfile**（✅ PR #3）
- 基础镜像 `python:3.12-slim`；**先复制依赖清单、安装依赖，再复制代码**：只改代码时，安装依赖那一层直接用缓存
- 镜像里只有"代码 + Python 依赖"。密钥（`.env.ocr`）运行时以只读方式挂载；向量库（`.rag_data/`）运行时挂载目录；Ollama 跑在宿主机上，通过环境变量 `EMBED_BASE_URL` 告诉服务它的地址
- `.dockerignore`：密钥、数据、Mac 的虚拟环境 `.venv`（里面是 macOS 的程序，在 Linux 容器里用不了）、测试，都不进镜像
- 服务监听 `0.0.0.0`：只监听 `127.0.0.1` 的话，容器外面访问不到

**在 Mac 上运行**

```bash
docker build -t rag-notes:dev .
docker run --rm -p 8000:8000 \
  -v "$PWD/.rag_data:/app/.rag_data" \
  -v "$PWD/.env.ocr:/app/.env.ocr:ro" \
  -e EMBED_BASE_URL=http://host.docker.internal:11434/v1 \
  rag-notes:dev
```

- 实测：镜像是 `linux/arm64`；容器里问答正常。**Mac 上（macOS 版 Milvus Lite）建的向量库，Linux 容器里也能直接读**，搬到别的机器上不用重新建库
- Docker Desktop 没打开时，会报 `failed to connect to the docker API ... docker.sock`：`docker` 命令只是客户端，真正干活的引擎要先启动

**CD**（✅ `.github/workflows/release.yml`）

```
git push origin v0.1.0 → Release workflow：① 测试（复用 ci.yml）→ ② 构建 amd64 + arm64 镜像 → 推送到 ghcr.io/sty-wq/rag
```

- **只有推送 `v` 开头的 tag 才发布**，平时推送代码只跑 CI。版本号用"语义化版本"：`主版本.次版本.修订号`，0.x 表示还在早期
- **发布前先测试**：`ci.yml` 加了 `workflow_call`，变成可以被调用的"可复用 workflow"，`release.yml` 直接调用它，测试步骤只写一份
- **一次构建两种架构**：GitHub 的机器是 amd64，用 QEMU 模拟 ARM CPU 构建 arm64。pull 时 Docker 会自动挑出适合本机的那一个。以后嫌慢，可以改用 GitHub 的原生 arm64 机器
- 登录 GHCR 用 GitHub 自动提供的 `GITHUB_TOKEN`，不需要自己配置密钥；workflow 里要声明 `packages: write` 权限
- 镜像标签：`v0.1.0` → `0.1.0`、`0.1`、`latest`（docker/metadata-action 自动生成）
- 镜像名必须全部小写，而仓库名是大写的 `RAG`，所以在 workflow 里用 bash 的 `${变量,,}` 转成小写
- 构建缓存用 GitHub Actions 的缓存（`type=gha`），依赖没变时会快很多
- GHCR 上新建的镜像默认是**私有**的，要在包的设置里改成公开，别人才能直接 pull
- 用到的 Action 版本（2026-10）：setup-qemu v4、setup-buildx v4、login v4、metadata v6、build-push v7

**第一次发布 `v0.1.0`**（2026-10-30）
- Release #1 全部通过，用时 7 分 21 秒：test 33 秒，image 6 分 43 秒（arm64 在 QEMU 模拟环境里构建，所以最慢）
- 部署就两条命令：

  ```bash
  docker pull ghcr.io/sty-wq/rag:0.1.0
  cd 项目目录   # 下面的 $PWD 要指向放着 .env.ocr 和 .rag_data 的目录
  docker run --rm -p 8000:8000 \
    -v "$PWD/.rag_data:/app/.rag_data" \
    -v "$PWD/.env.ocr:/app/.env.ocr:ro" \
    -e EMBED_BASE_URL=http://host.docker.internal:11434/v1 \
    ghcr.io/sty-wq/rag:0.1.0
  ```

- `docker buildx imagetools inspect ghcr.io/sty-wq/rag:0.1.0` 能看到一个 tag 下有 amd64、arm64 两个镜像；Mac 上 pull 时 Docker 自动挑了 arm64
- 踩过的坑：
  - 推送 tag 要单独执行 `git push origin v0.1.0`，普通的 `git push` 不会推送 tag
  - GHCR 上新建的包默认是私有的，要在 Package settings 里改成 Public
  - **在主目录下运行了 `docker run`**：`$PWD` 指错了地方，`-v` 找不到源文件时**不报错，而是自动建一个同名的空文件夹**，结果容器里的 `.env.ocr` 变成了空文件夹，服务启动失败。正式部署时改用 `--mount type=bind,src=…,dst=…`，源文件不存在会直接报错
  - `docker pull` 和当前目录无关；`docker build .` 和 `-v "$PWD/…"` 都和当前目录有关

**小修小补**（✅ 2026-10-30，分支 `chore/small-fixes`）
- ✅ `actions/checkout` v4 → v6、`actions/setup-python` v5 → v6：Release 日志提醒旧版本用的 Node.js 20 已经过时，v6 改用 Node.js 24，用法不变。`ubuntu-latest` 以后会换成 Ubuntu 26，先不管
- ✅ `llm.py`：配置文件不存在、或者是个文件夹时，直接提示"找不到配置文件"并点名可能是 `-v` 挂载错了，不再误报"缺少配置"。加了两个回归测试
- ✅ 问答提示词加了一条：资料相互矛盾时，不要混在一起，要明确指出矛盾，并分别标出处（见第 8 节）

**停车场：Kubernetes、分布式部署、按需扩展、滚动更新**（2026-10-30 新增）

现在还做不了，原因不在 K8s，而在于我们的服务还不满足前提条件：
- **Milvus Lite 是单个文件，同一时间只允许一个进程打开** → 服务没法起多个副本，扩展不了。要先换成 Milvus 服务器版（standalone）
- **服务不是"无状态"的**：数据在本地文件里。K8s 的扩缩容和滚动更新，要求每个副本都能随时被删掉、再新建
- embedding（Ollama）应该拆成一个独立的服务，单独扩展
- 资源：哪怕是精简版的 k3s 也要占几百 MB 内存；"分布式"至少要两三台机器
- 没有流量，"按需扩展"就没有意义，也拿不出数据来说明效果

建议的顺序：CD → 用 Docker 部署到服务器 → 评测集 → 混合检索 / rerank → 换成 Milvus 服务器版、服务改成无状态 → 再上 K8s（可以先用 k3s 学）

**停车场：评测数据集**（2026-10-30 新增，部署之后、混合检索和 rerank 之前做）
- **自己的评测集为主**：20～50 个问题，标好"应该找到哪几块"；有意混入"资料里没有答案"的问题（测拒答）和"换了说法"的问题（避免只靠字面重合）。可以让 LLM 针对每个 chunk 出题，再人工检查。公开数据集测不到我们的解析、结构修正和切分
- **C-MTEB 的检索任务**（DuRetrieval、T2Retrieval 等）：用来比较 embedding 模型、混合检索、rerank
- **RGB**（AAAI 2024，有中文）：测抗噪声、**拒答**、信息整合、**识别资料中的错误**，正好对应上面遇到的两个情况
- **CRUD-RAG**（中文）：按"增删改查"四类任务评测，语料是新闻

---

## 参考资料

- [Is Semantic Chunking Worth the Computational Cost?（NAACL 2025 Findings）](https://arxiv.org/abs/2410.13070)：语义切分带来的提升不稳定，成本却高不少
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)：top-20 检索失败率降低 49%
- [LangChain: Multi-Vector Retriever](https://python.langchain.com/docs/how_to/multi_vector)：摘要索引的参考实现
- [阿里云百炼：Embedding 模型规格](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)
- [硅基流动：Embeddings 接口与各模型输入上限](https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings)
- [Milvus Lite（PyPI）](https://pypi.org/project/milvus-lite)：本地文件版 Milvus，支持的平台和限制
- [Ollama 上的 embedding 模型一览](https://www.morphllm.com/ollama-embedding-models)：qwen3-embedding 各尺寸的维度、上下文和体积，以及 `/v1/embeddings` 接口
- [CRUD-RAG：中文 RAG 评测基准](https://arxiv.org/abs/2401.17043)
- [RGB：Benchmarking LLMs in Retrieval-Augmented Generation（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/29728)
- [C-MTEB（PyPI）](https://pypi.org/project/C-MTEB/)
- [DeepSeek API: Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [DeepSeek API: JSON Output](https://api-docs.deepseek.com/guides/json_mode)
