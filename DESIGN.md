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
  - **给没有结构的文档"赋予结构"**（2026-11-01 新增）：纯文本没有标题，现在只能按长度切。让 LLM 读全文，输出"在第几段前面插入一个几级标题、标题叫什么"
    - 沿用结构修正的原则：LLM **只出方案、不改原文**，代码负责插入标题，插完做无损检查
    - 长文档要分窗口读；先做规则（疑似标题），规则搞不定的再交给 LLM，控制费用
    - **评测不用另找数据**：把 Qasper 论文的标题全部去掉，变成纯文本，做三组对比：原始标题（上限）/ 没有标题（下限）/ LLM 补的标题。还能直接对比 LLM 补的标题和原始标题有多像
    - 相关工作：HiChunk / HiCBench（腾讯优图，arXiv 2509.11552，专门评测从纯文本推断层级结构）、LumberChunker

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
| `RAG_MILVUS_URI` | `.rag_data/milvus.db` | 项目专用配置；以 `.db` 结尾就是 Milvus Lite，`http://地址:19530` 是 Milvus 服务器 |

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

---

## 10. 评测（🚧，`evaluation/`）

**决定**（2026-10-30）：**只用公开数据集**（没有自己的笔记可用，而且公开数据别人能复现）。评测分三层，每层回答一个问题：

| 层 | 回答什么问题 | 数据 | 状态 |
|---|---|---|---|
| **C. 检索和问答** | 切分、混合检索、rerank 能不能提高准确率 | CRUD-RAG | 🚧 检索部分已实现 |
| **B. 格式对照实验** | 文件格式会不会影响最终效果 | 同一批 CRUD-RAG 新闻，分别做成 txt / docx / PDF / 扫描件，题目不变 | 📌 |
| **A. 解析层** | 每种格式解析得对不对 | 程序生成的已知原文的文档（md/txt/docx/PDF，含 GBK）；真实扫描件用 OmniDocBench 的中文页面 | 📌 |

顺序：C → B → A（B 直接复用 C 的题目，A 工作量最大）

### 10.1 CRUD-RAG 小样本（`evaluation/build_crud.py`）

- 数据结构（实测）：`split_merged.json` 里 `questanswer_1doc` 800 条、`2docs` 797 条、`3docs` 797 条；每条有 `ID`（事件 id）、`event`、`news1`～`news3`（题目依据的新闻原文）、`questions`、`answers`；语料 48 个文件，**一行一篇新闻**，共约 9 万行
- 抽样规则（固定种子 42，每次结果一样）：
  - 每类 100 道题，**同一个事件只用一次**：三类题目会围绕同一事件出题，否则一道题的标准答案文档会被另一道题的文档抢排名
  - 题目依据的新闻 = 标准答案文档；文件名是正文的 md5，**标注和切分方式无关**，改切分也不用重新标注
  - 3000 篇随机新闻作为干扰项，**排除标准答案的副本**（语料库里同一篇新闻可能有好几份，用正文中间的 3 小段文字判断）
  - 50 道**无答案题**：抽 1doc 题，但故意不把原文放进语料
- 数据不提交到 git（`evaluation/data/` 已忽略），只提交脚本和汇总结果
- 已知局限：干扰项里可能有**同一事件的其他报道**（不是副本，但也能回答问题），会让分数略微偏低；新闻是纯文本，测不到标题结构相关的处理

### 10.2 检索评测（`evaluation/run_retrieval.py`）

- 不调用 DeepSeek，**不花钱、结果可复现**；用单独的向量库文件，不碰笔记的库
- 指标（k 指"前 k 个 chunk"，因为交给 LLM 的就是前 k 个 chunk）：
  - **Hit@k**：前 k 个里有没有至少一篇标准答案文档
  - **Recall@k**：前 k 个覆盖了多少比例的标准答案文档（2docs / 3docs 题只找到一部分也会体现出来）
  - **MRR**：第一个找对的结果排名的倒数，再取平均
  - **相似度门槛分析**：按 `MIN_SCORE` 算"有答案的题被误拦的比例"和"无答案的题被拦下的比例"，用来定门槛
- 结果：汇总存到 `evaluation/results/<时间>_<标签>.json`（提交到 git，留作历史记录，里面记着 git 提交号、模型、切分参数），每道题的明细存到 `evaluation/data/crud/runs/`（不提交）
- 指标函数和抽样规则都有离线测试（`tests/test_evaluation.py`），CI 里会跑

### 10.3 结果记录

| 日期 | 标签 | 提交 | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | v0.1.2 基线（原题，3000 篇随机干扰） | c987e9f | 0.920 | 0.993 | 0.947 | 0.977 | 0.952 |
| 2026-10-30 | v0.1.2 基线（**难版 v1**，3000 篇随机干扰）⚠️ 见下方说明 | a95465c + 改写脚本 | 0.843 | 0.980 | 0.916 | 0.956 | 0.904 |
| 2026-10-30 | v0.1.2 基线（难版 v2，3000 篇随机干扰） | 3a5b2bc + 改写 v2 | 0.867 | 0.977 | 0.926 | 0.973 | 0.912 |
| 2026-10-31 | v0.1.2 基线（易版，**1.5 万篇**干扰） | 55cc2c1 | 0.853 | 0.977 | 0.900 | 0.937 | 0.907 |
| 2026-10-31 | v0.1.2 基线（难版 v2，**1.5 万篇**干扰）⚠️ 标注有噪声，见下方 | 55cc2c1 | 0.767 | 0.930 | 0.849 | 0.909 | 0.841 |

⚠️ **人工抽查发现的问题**：难版 v1 的改写提示词允许"只保留最主要的一两个方面"，于是 2docs / 3docs 的题会被**删掉一问**（例如"美国对台军援包括哪些武器、值多少钱，**以及中国官方如何回应**" → 改写后丢了最后一问）。被删掉的那一问往往正好对应其中一篇新闻，那篇新闻就不再被题目需要，**Recall 被冤枉地拉低了**。Hit@k 和 MRR 不受影响（只要找到任意一篇标准答案文档就算）。已改成 v2 提示词：每一问都要保留。这就是人工抽查的价值：分数下降不一定是"题目变难了"，也可能是"题目出错了"
- 抽查中其他 6 道改写都自然、意思没变，比如"龙贸通平台是哪家公司开发的？什么时候上线试运行？"

**难版 v2**（350 道全部改写成功；平均长度 50 字 → 26 字；和原文的字面重合度 0.527 → 0.455）
- 抽查确认多问的题每一问都保留了，例如"美国对台军援都包括啥武器？值多少钱？中国怎么回应的？"
- 和 v1 相比，3docs 的 Recall@10 从 0.937 回到 0.953，印证了"v1 删掉一问会冤枉 Recall"的判断
- 和易版相比：**Hit@1 −5.3、MRR −4.0**，Hit@5 −1.6。提升空间主要在排序上
- 字面重合度只降到 0.455：改写保留了"龙贸通""岸田"这类关键名称，这也符合真实用户的提问习惯
- ⚠️ v1 和 v2 之间约 1 个点的差别，在 300 道题上属于正常波动范围，不能过度解读。以后比较两个版本时，要逐题对比"变好几道、变差几道"
- **以后的基线以"难版 v2"为准**，易版作为参考一起报告

**干扰项加到 1.5 万篇**（2026-10-31；抽样时排除了 1282 篇标准答案的副本，比例约 8%；共 15593 篇、20724 块；建库用时 1548 秒，约 13 块/秒）
- 终于有了提升空间：难版 Hit@1 0.767、Hit@5 0.930、Recall@5 0.849、MRR 0.841；3docs 的 Recall@5 只有 0.753
- 门槛：有答案 top1 中位数 0.738，无答案 0.630，靠得更近了

**⚠️ 错误分析：没找到的题里，有不少其实"找对了"**（难版 Hit@5 = 0 的共 21 道：1doc 10、2docs 4、3docs 7；人工看了前 6 道 1doc）

| 题目 | 检索结果 | 判断 |
|---|---|---|
| 惠誉把美国评级从哪级降到哪级？ | 前 4 名都是**同一事件的其他报道**，第 3、4 名原文就写着"从 AAA 下调至 AA+" | ❌ 标注漏了，其实答得出 |
| 双一流高校每年选拔多少理工科本科生读研？ | 第 1 名是人民网的同题报道，和标准答案文档（排第 6）有大段相同的文字，答案原句一字不差。"排除副本"没抓到它，是因为一个用半角括号"("、一个用全角"（" | ❌ 标注漏了 + 去重漏网 |
| 俄非峰会在哪个城市、哪天开？ | 第 2、3 名都写着"7 月 27 日在圣彼得堡开幕" | ❌ 标注漏了（顺便发现：参考答案把年份写成了 2021，数据集本身也有错） |
| 特朗普下周可能因为啥又被起诉？ | 第 3 名是后续报道"因企图推翻佐治亚州选举结果被起诉" | ⚠️ 半对 |
| 尼日利亚停了给尼日尔的啥供应？ | 前 4 名都是尼日尔政变的其他报道，都没提到停电 | ✅ 真没找到 |
| 台风会商部署了哪些省份？ | 都是**别的日期**的防汛会商（8 月 13 日、19 日），话题一样、事件不同 | ✅ 真没找到，这是有价值的"难干扰项" |

结论：
- **"一道题只有原文那几篇算对"的标注方式，在干扰项变多以后会漏掉很多其实相关的文档**，绝对分数被低估了。语料里同一事件往往有好几篇报道
- 对比两个系统时，噪声对双方都有，相对比较大体还能用；但有可能**冤枉某些方法**：比如 rerank 把另一篇也能回答问题的报道排到第一，会被算成"变差了"
- 修复计划：
  1. **补充相关性标注（TREC 式 pooling）← 主要修复**：把各个系统检索到的前 k 篇汇总起来，让 LLM 判断"这篇文档能不能回答这道题"，能回答的也算对。判断结果缓存起来，新系统只需要判断新出现的文档；再人工抽检一部分，确认 LLM 判断得准。这是信息检索领域补全标注的标准做法
  2. **去重前统一全角/半角**（`unicodedata.normalize("NFKC", …)`）：试算了一下，会多抓出 214 篇干扰项，但只涉及 8 篇标准答案文档、7 道题；21 道没找到的题里只有 1 道和它有关。影响小，而且 214 篇里可能有因为"举报/反馈"这类套话误伤的，要先看一眼再决定
- 其他：建库 26 分钟太慢，可以试试调大 `EMBED_BATCH_SIZE`；建库时日志里的 gRPC `too_many_pings` 是向量库连接在漫长的向量化期间一直空闲导致的，无害，以后可以改成"先向量化、再连接向量库"

**⚠️ 切块方式让这套题偏简单**（2026-10-31，自己发现的）
- 新闻都很短：语料长度中位数 680 字，而 `MAX_CHUNK_CHARS = 800`，**70% 的新闻整篇就是一块**
- 标准答案文档更短：中位数 409 字（干扰项是 688 字），**95% 整篇就是一块**，剩下 5% 是两块
- 所以现在测的其实是"在一堆新闻里找到对的那一篇"，不是"在长文档里找到对的那一段"。题目是照着整篇新闻出的，整篇又正好装在一块里，这一块和题目自然很像
- 切块策略基本没被测到：改 `MAX_CHUNK_CHARS`、换切分方式，在这套题上很可能看不出差别（这是 B 层要解决的问题）
- Hit@k 是按文档算的：前 k 块里只要有标准答案文档的任意一块就算对，不检查找到的那一块里到底有没有答案。新闻这么短时没区别，换成长笔记就会偏高
- 待查：标准答案文档为什么比干扰项短这么多（CRUD-RAG 出题时挑的新闻就短？还是别的原因）；长度差别会不会让检索变容易

### 10.5 评测一个 RAG 系统要看哪些指标（全景）

| 层 | 指标 | 我们的现状 |
|---|---|---|
| 入库 | 解析准确率（编辑距离）、结构保留、失败率、chunk 长度分布 | ❌（计划中的 A 层） |
| 检索 | **Hit@k / Recall@k、MRR**、nDCG@k、Context Precision（前 k 块里有多少有用） | ✅ Hit / Recall / MRR |
| 生成 | **正确性**（LLM 裁判对照参考答案）、**忠实度 / 幻觉率**（每句话在资料里有没有依据）、答案相关性、**引用准确率**、**拒答率 / 误拒率**、多问题的完整性 | ❌ 下一块要补 |
| 系统 | **延迟**（P50 / P95，检索和 LLM 分开）、**成本**（token / 次）、建库耗时、错误率 | ❌ 很容易加 |
| 鲁棒性 | 换问法、错别字、更多干扰、资料矛盾、文档里的恶意指令 | 🚧 难版题目 |

- **按层定位问题**：每道答错的题，先看"正确资料找到没有"。找到了还答错 → 生成层的问题；没找到 → 检索层的问题
- **LLM 裁判也要验证**：先人工标几十道，确认 LLM 打分和人工一致，才信它的分数
- 现成工具（Ragas、TruLens 等）的原理相同；简历上最有分量的五个：Recall@k / MRR、正确性、幻觉率、拒答率、延迟和成本

**难版 vs 易版**（只改了题目的问法，语料和系统都没变）：
- Hit@1 下降 7.7 个点（0.920 → 0.843），MRR 下降 4.8 个点：**口语化的短问题主要伤的是"排序"**，正确文档还在前 5 名，但不一定排第一
- Hit@5 只降了 1.3 个点（0.993 → 0.980），仍然接近天花板
- 1doc 降得最多：Hit@1 0.94 → 0.83，Hit@5 1.00 → 0.96（有 4 道题前 5 名都没找到）
- 门槛：有答案的 top1 中位数 0.788 → 0.710，无答案的 0.596 → 0.569，两者越来越近，0.35 依然一道都没拦下
- 结论：**rerank 的主战场是 Hit@1 / MRR**；Hit@5 这一项需要更大的干扰项才能拉开差距

**基线的分析**（4685 块；按类型：1doc Recall@5 = 1.000，2docs 0.955，3docs 0.887）

1. **太简单了，几乎没有提升空间（天花板效应）**：Hit@5 已经 0.993，剩下的提升空间只有不到 1 个点。300 道题里，一道题就是 0.33 个点，再做优化也很难和随机波动区分开。原因：
   - 题目是 GPT-4 照着新闻原文出的，又长又具体，带着日期、机构全称、引号里的名称，**和原文的字面重合度很高**；真实用户的提问通常更短、更口语化
   - 干扰项是随机抽的，话题大多和题目八竿子打不着
2. **相似度门槛 0.35 完全没起作用**：有答案的题 top1 中位数 0.788，无答案的题 0.596，**无答案的题一道都没被拦下**，全部交给了 LLM 去判断。0.35 是当初只看一个例子定的，这里被数据否定了
3. 真正还有空间的：**3docs 的 Recall@5（0.887）** 和 **Hit@1 / MRR**（排序）

**下一步：先把评测集变难，再做优化**（否则优化的效果测不出来）
- ✅ **"难版"题目**（`evaluation/rewrite_questions.py`）：让 LLM 把每道题改写成简短、口语化的问法，标准答案不变，所以不会引入标注错误；原题保留为"易版"，两套一起报。改写时 LLM 只看题目、不看原文；结果有缓存；生成 `review_hard.md` 对照表供人工抽查，并统计改写前后和原文的字面重合度（"两字组"重合比例）
- **扩大干扰项**：从 3000 篇增加到上万篇
- 门槛：从明细里算不同门槛下的误拦率和拦对率；**用一个种子的样本来定门槛，再换一个种子的样本来验证**，避免对着测试集调参
- 汇报差别时，按题目逐一对比"变好了几道、变差了几道"，而不只看平均分

### 10.6 Qasper：长文档评测集（`evaluation/build_qasper.py`）

为什么要第二套：CRUD-RAG 的新闻太短（见 10.3"切块方式让这套题偏简单"），测不到切块和章节结构，而这正是我们项目的重点。

- 数据：Qasper test 部分，416 篇 NLP 论文全文，1451 道题。英文，原始数据从 Hugging Face 下载 parquet（`allenai/qasper`），放在 `evaluation/data/qasper/raw/`
- 论文 → Markdown：标题 `#`，摘要和章节 `##` / `###`。子章节在原数据里写成 `Model ::: Background`；父章节自己没有正文时原数据里不出现，转换时补上它的标题。行首的 `#` 要转义
- **416 篇放进同一个库**，别的论文都是干扰项。按现在的切法共 17885 块，每篇中位数 41 块，块长度中位数 570 字
- embedding：OpenAI `text-embedding-3-small`（云端）。Qasper 的所有实验都用它，和 CRUD 的本地模型分数不能直接比
- 题目分四类（全部保留）：

| 类型 | 题数 | 含义 | 怎么用 |
|---|---|---|---|
| answerable | 1309 | 至少一个标注者给了文字证据 | 算检索分 |
| float_only | 42 | 证据全是图表，数据里没有表格内容 | 应该拒答。单独打标签：以后能解析表格了，它们会变成"应该能回答"，可以用来测解析的改进（A 层） |
| unanswerable | 79 | 所有标注者都说没有答案 | 应该拒答 |
| no_evidence | 21 | 有答案但没标证据 | 算不了检索分，留给生成质量评测 |

- **按证据算分**：一块"找对了" = 块里完整包含某个证据段落，或者这块就是某个长证据段落切出来的一部分（至少 50 字）。证据只保留和正文段落一字不差的（忽略空白）；有 222 条证据其实是章节标题，丢掉
- Recall 在几个标注者里**取最好的那一个**（Qasper 官方也是这样），不取并集
- 额外记录 `doc_hit@k`：前 k 块里有没有来自对的那篇论文。`doc_hit` 高、`hit` 低，说明论文找对了、段落没找对
- 上限检查：1309 道题的证据，全部能被它所在论文的至少一块覆盖（切块没有把证据切丢）
- ⚠️ 相似度门槛 0.35 是按本地 qwen 模型定的，OpenAI 模型的分数范围不一样，用 `--min-score` 另外定

**第一次跑（2026-11-01，3-small，建库 1032 秒）：原题直接在整个库里检索 → 失败的设计**
- hit@5 0.238，连"找到对的论文"的 doc_hit@5 都只有 0.360
- 原因：Qasper 的题是拿着某一篇论文问的（"what was the size of the datasets?"），题目里不说是哪篇。416 篇放进一个库，这种题从一开始就没法找。**测的不是我们的系统**
- 也不能用"只在这道题所属的论文里检索"（`--within-doc`）来修：那是假设已经知道是哪篇，真实使用时做不到

**改法：给题目补上"是哪篇论文"**（`evaluation/contextualize_qasper.py`）
- LLM 只看问题（不看标题、正文、证据），判断"能不能看出问的是哪一篇"
- 看不出的，用固定模板补标题：`In the paper "<标题>", <原问题>`。不让 LLM 改写，题目的意思一个字都不会变
- 结果：1451 道里只有 16 道能看出是哪篇，**99% 都要补标题**。抽查的 8 道判断都合理。DeepSeek 调用 1451 次，输入约 30 万、输出约 6 万 token

| 日期 | 标签 | 题数 | hit@1 | hit@5 | recall@5 | recall@10 | mrr | doc_hit@5 |
|---|---|---|---|---|---|---|---|---|
| 2026-11-01 | 原题，整个库（❌ 设计有误，仅作对照） | 1309 | 0.114 | 0.238 | 0.215 | 0.274 | 0.167 | 0.360 |
| 2026-11-01 | 补标题，整个库 | 1309 | 0.255 | 0.585 | 0.537 | 0.721 | 0.399 | 0.999 |
| 2026-11-01 | **修复英文断句（补标题，整个库）← 以后以这一行为准** | 1309 | 0.261 | 0.619 | 0.560 | 0.723 | 0.408 | 0.999 |

**修复英文断句**（`chunks.split_sentences` / `hard_cut`）：原来"按句子切"只认中文标点和 `!?;`，不认英文句号，超过 800 字的英文段落都在第 800 个字处硬切，约 2780 处把单词从中间切断（"automated appr | oach"）。现在英文句号要满足"句号 + 空格 + 大写字母"才切，跳过 e.g.、et al.、Fig. 这类缩写；只能硬切时也切在空格处。修完以后切断单词的地方是 0，CRUD 的块数不变

逐题对比（符号检验：只看变好、变差的题，像抛硬币一样检验"变好的比变差的多"是不是偶然）：

| 指标 | 变好 | 变差 | p 值 | 结论 |
|---|---|---|---|---|
| hit@1 | 51 | 43 | 0.47 | 看不出差别 |
| **hit@5** | **69** | **25** | **< 0.001** | **确实变好了（+3.4 个点）** |
| hit@10 | 28 | 20 | 0.31 | 看不出差别 |

- 平均分只差几个点，但逐题看，hit@5 变好的题是变差的近 3 倍，不是偶然
- 有变差的题很正常：块的边界变了，有些原来碰巧排在前面的块被挤下去了

**候选池深度**（2026-11-02，`--k-max 50`，不用重建库）：Recall@10 0.723 → **@20 0.898 → @50 0.986**
- 证据几乎都捞得到，只是排得不够靠前 → **先做 rerank**，混合检索往后放
- ⚠️ @50 偏乐观：每篇论文中位数只有 41 块，又带着标题检索，前 50 名基本就是整篇论文。换成更大的笔记库，候选池要重新量
- ⚠️ MRR 从 0.408 变成 0.421，不是变好了：k_max=10 时排在第 10 名以后的算 0 分，k_max=50 时算 1/11～1/50。**比较 MRR 时 k_max 必须一样**

- **doc_hit@5 = 0.999**：带上标题以后，找到对的论文几乎不成问题（每一块的 `embed_text` 里都带着论文标题）
- 所以剩下的分数测的就是"**在一篇长论文的约 41 块里找到对的那一段**"：hit@1 只有 0.255，recall@10 0.721，提升空间很大。这正是我们想要的切块 / 排序评测
- self 组只有 15 道，太少，不单独解读
- 相似度门槛完全没用：该拒答的 121 道 top1 中位数 0.76，有答案的 0.78。论文里没有答案，可同一篇论文的段落和问题照样很像，**靠检索分数判断不了"该不该拒答"，要到生成阶段判断**

### 10.4 以后（📌）

- 问答层：用 DeepSeek 对照参考答案打分；无答案题统计拒答率
- **RGB**（AAAI 2024，有中文）：测抗噪声、**拒答**、信息整合、**识别资料中的错误**
- **C-MTEB 的检索任务**（DuRetrieval、T2Retrieval 等）：用来比较 embedding 模型
- 更难的干扰项：同一事件的其他报道

---

## 参考资料

- [Is Semantic Chunking Worth the Computational Cost?（NAACL 2025 Findings）](https://arxiv.org/abs/2410.13070)：语义切分带来的提升不稳定，成本却高不少
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)：top-20 检索失败率降低 49%
- [LangChain: Multi-Vector Retriever](https://python.langchain.com/docs/how_to/multi_vector)：摘要索引的参考实现
- [阿里云百炼：Embedding 模型规格](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)
- [硅基流动：Embeddings 接口与各模型输入上限](https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings)
- [Milvus Lite（PyPI）](https://pypi.org/project/milvus-lite)：本地文件版 Milvus，支持的平台和限制
- [Ollama 上的 embedding 模型一览](https://www.morphllm.com/ollama-embedding-models)：qwen3-embedding 各尺寸的维度、上下文和体积，以及 `/v1/embeddings` 接口
- [CRUD-RAG：中文 RAG 评测基准](https://arxiv.org/abs/2401.17043)（数据：[GitHub](https://github.com/IAAR-Shanghai/CRUD_RAG)）
- [OmniDocBench（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)：981 页真实 PDF 页面，中英文、9 种文档类型，有标准文字标注
- [RGB：Benchmarking LLMs in Retrieval-Augmented Generation（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/29728)
- [C-MTEB（PyPI）](https://pypi.org/project/C-MTEB/)
- [DeepSeek API: Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [DeepSeek API: JSON Output](https://api-docs.deepseek.com/guides/json_mode)

## 11. 混合检索与模型重排（2026-11-02）

共享流程在 `retrieval.py`：问题 → dense / BM25 两路召回 → 按 chunk_id 用 RRF 融合 → 可选 CrossEncoder 重排 → 最终 top-k。
`search.py`、`answer.py` / `api.py`、`evaluation/run_retrieval.py` 共用同一流程，保留 dense、BM25、hybrid 三种模式，便于做对照实验。

### 11.1 BM25 与候选池

- 数据库地址使用项目专用 `RAG_MILVUS_URI`，优先级为构造函数 uri 参数 > RAG_MILVUS_URI > 旧 MILVUS_URI > 默认笔记库。PyMilvus 导入时会把 MILVUS_URI 按服务器 URL 解析；旧值为本地 `.db` 路径时，本项目会在 SDK 导入前将它迁移到 RAG_MILVUS_URI，并在当前 Python 进程中清空 MILVUS_URI，兼容旧命令及 `.env` 配置。已有的 RAG_MILVUS_URI 优先，旧服务器 URL 保留原行为；空配置使用默认库。终端中的变量不受子进程迁移影响。
- `lexical.py` 实现 BM25 Okapi 倒排索引。英文按单词和标识符切分，中文使用单字和相邻双字，统一 NFKC 与大小写；没有语义分词或词干处理。
- `VectorStore.lexical_search()` 用 query_iterator 分批读取已有集合的 chunk 文字和出处，在进程内建立索引，**不修改 Milvus schema、不重算已有 embedding**。
- 索引输入是标题路径 + 库中实际存储的原文；首次查询加载，后续复用缓存。`rebuild()` 或本项目的索引元信息变化会使缓存失效。绕过本项目直接写集合不在缓存一致性保证内。
- 当前适合单进程、万级 chunk。每个服务进程有自己的 BM25 索引，启动耗时和内存随语料增加；大规模场景应另外评测持久化全文索引。
- RRF：`sum(1 / (rrf_k + rank))`，默认 `rrf_k=60`。先按 chunk_id 去重，保留两路原始分数；不把 BM25 与余弦分数直接相加。
- `candidate_k=50` 表示每路最多召回 50 块，融合后最多取 50 块送去重排；`k=5` 是最终交给问答的数量。若 k 大于 candidate_k，候选池会相应扩大。
- doc_name 过滤同时作用于两路；仅 BM25 模式不调用 query embedding，也不要求当前 embedding 模型与建库模型相同。

### 11.2 重排配置

`reranker.py` 提供本地和 HTTP 两种真实模型接口。默认关闭，启用后失败会报错，不静默降级为未重排结果。

本地依赖和模型：

```bash
.venv/bin/python -m pip install -r requirements-rerank.txt
.venv/bin/python search.py "如何避免重复扣费" --mode hybrid --candidate-k 50 --rerank local -k 5
```

默认模型 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，显式启用后首次加载并缓存到 `.rag_cache/models/`，不需要云端 key。模型为 query-document 对打分，不改写原文。默认最大输入长度 512 token，长候选会由模型分词器截断；这是当前限制，需要对长代码另做实验。

环境变量示例见 `retrieval.env.example`。本地主要配置：`RERANK_MODEL`、`RERANK_DEVICE`、`RERANK_REVISION`、`RERANK_CACHE_DIR`、`RERANK_BATCH_SIZE`、`RERANK_MAX_LENGTH`、`RERANK_TIMEOUT`。模型在独立常驻子进程 `_rerank_worker.py` 中加载和复用，推理串行化以限制并发内存；这样避开 macOS 上 FAISS 与 PyTorch 的 OpenMP 冲突，不需要额外 Web 服务。API 关闭时释放子进程；超时或崩溃时报错并清理，后续请求可以重新启动。

HTTP 模式设置 `RERANK_BACKEND=http`、`RERANK_BASE_URL`、`RERANK_MODEL`，需要时再设置独立的 `RERANK_API_KEY`。客户端 POST 到 `<BASE_URL>/rerank`，请求包含 model / query / documents / top_n，响应需要为 `results: [{index, relevance_score}]`，且返回全部候选的唯一 index。它不是 OpenAI 标准接口；接入具体供应商时要核对该协议。

### 11.3 API 与分数

```bash
RAG_RETRIEVAL_MODE=hybrid RERANK_BACKEND=local .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

`POST /ask` 支持可选的 `mode`、`candidate_k`、`rerank`，例如：

```json
{"question": "如何避免重复扣费", "k": 5, "mode": "hybrid", "candidate_k": 50, "rerank": true}
```

未传字段时采用服务配置；API / search CLI 默认 hybrid，重排由 `RERANK_BACKEND` 控制，默认 none。Python 的 `run_search()` 和评测 CLI 默认保留 dense 基线。请求启用重排但服务没有配置时返回 503，可显式传 `rerank:false` 做对照。`GET /health` 返回检索配置和重排模型信息。

返回出处增加 `score_type`、`dense_score`、`lexical_score`、`rrf_score`、`rerank_score`。`score` 是最终排序分数，不同 score_type 的数值不能直接比较。原 `RAG_MIN_SCORE=0.35` 只用于未重排的 dense 余弦分数；BM25 和 RRF 不套用此门槛。模型重排默认也不按分数拒答，如需门槛，用开发集校准 `RAG_MIN_RERANK_SCORE`。引用与资料充分性仍由现有问答流程处理，新增重排本身不证明回答正确。

Docker 默认镜像支持 BM25 / hybrid / HTTP 重排；需要镜像内本地模型时，使用 `docker build --build-arg INSTALL_LOCAL_RERANK=true`，模型缓存建议单独挂载。

### 11.4 对照评测

已有索引可直接复用，不需要为 BM25 重建向量库。以下命令在相同 questions / embedding / corpus 上对比；使用 Qasper 时要配置与其索引相同的 embedding 模型和地址。

```bash
.venv/bin/python -m evaluation.run_retrieval --label "dense 对照" --reuse-index --mode dense
.venv/bin/python -m evaluation.run_retrieval --label "BM25 对照" --reuse-index --mode bm25
.venv/bin/python -m evaluation.run_retrieval --label "hybrid 对照" --reuse-index --mode hybrid --candidate-k 50
.venv/bin/python -m evaluation.run_retrieval --label "hybrid 加重排" --reuse-index --mode hybrid --candidate-k 50 --rerank local
```

汇总保留原 Hit / Recall / MRR，同时记录检索方式、候选池、RRF 参数、重排模型和版本。逐题明细增加 chunk_id 和各阶段分数，便于核对变好/变差的题。MRR 比较时固定 k_max。没有配置对应分数门槛时 gate 标记 disabled，不能把它解释成拒答能力。检索耗时包含首次 BM25 构建/模型加载，不包含批量 query embedding；冷启动和稳定延迟需要分开看。

自动化测试仍全部离线，模型与 HTTP 用假依赖；Milvus 测试使用临时真实数据库，覆盖旧 schema、跨分页加载、过滤、缓存失效、融合、重排协议和问答/API 的分数处理。真实模型小样例或有限样本检查只证明接线可用；完整数据上的收益必须由正式对照实验报告。

### 11.5 Qasper 重排回退分析与问题分离（2026-11-02）

#### 实验现象

在同一份 `questions_ctx.jsonl`、17887 块索引、`text-embedding-3-small`、`k_max=10` 下，候选池固定 50。本地重排模型为 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，revision 为 `1427fd652930e4ba29e8149678df786c240d8825`。

| 实验 | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 | 检索 P50/ms |
|---|---|---|---|---|---|---|
| A：Dense | 0.2613 | 0.6188 | 0.5597 | 0.7230 | 0.4083 | 4.0 |
| C：Hybrid | 0.1322 | 0.5088 | 0.4398 | 0.6543 | 0.2941 | 17.0 |
| D：Dense + 本地 rerank | 0.2254 | 0.6960 | 0.6340 | 0.8166 | 0.4189 | 237.7 |
| E：Hybrid + 本地 rerank | 0.2261 | 0.6952 | 0.6335 | 0.8169 | 0.4192 | 247.4 |

来源是 `evaluation/results/` 下标签前缀 `qasper_ab_20261102_171451_` 的 A/C/D/E 结果；A、D 逐题明细分别为 `evaluation/data/qasper/runs/20261102-171638.jsonl`、`20261102-173333.jsonl`。这里的 P50 不含 query embedding 和答案生成；不能称为端到端延迟。以上是修改前的真实全量结果，不是下面新方案的成绩。

结论：重排改善了前 5/10 条证据覆盖，但第一名命中下降；等权混合检索没有体现额外收益。不能写成“所有排序指标均提升”，也不能把这套英文论文结果直接推广到中文 CRUD。

#### 原第一名命中、重排后未命中的 206 道题

全量机械检查 206 道，定向人工复核 26 道（并非随机样本，不能估计各原因的总体比例）：

- 204 道的 D 第一名仍来自正确论文；104 道变为摘要，30 道为引言，33 道为结论类章节。
- 166 道的首个标注证据落在第 2～5 名，31 道落在第 6～10 名，9 道前十未命中。多数是论文内的段落排序回退。
- 175 道的 D 第一名原本就在 A 前十，因此单纯缩小 50 个候选不能解决大多数回退。
- 原 A 第一名和 D 第一名组成 412 对输入，实际 tokenizer 长度最大 464，全部低于 512；本批相对次序反转不是输入截断导致的。
- 同一模型离线回放复现了 206 对的排序关系。D 第一名分数最大误差约 0.0000215；仍留在 D 前十的 180 个原 A 第一名，分数最大误差约 0.0000143。没有发现支持分数错位的证据。

已确认的问题有三类，可能重叠：

1. **问题中的论文标题影响重排偏好。** 保持候选文字与模型不变，仅把 `In the paper "<title>", <question>` 换为显式保存的原问题，105/206 对中原 A 第一名重新超过 D 第一名。只去候选标题路径翻转 23 对，两者都去掉翻转 94 对。这里比较的是两个候选，不能解释成“完整 Top-50 已修复 105 题”，更不能当作全量新指标。
2. **主题相关不等于回答了细节。** `1906.08871-q01` 的具体 EEG 五种特征被只说使用 EEG 的摘要挤下；`1903.02930-q02` 问训练数据规模，模型把网络层数/batch size 放在 64M 视频片段之前；`2002.04326-q01` 问数据集规模，模型把选项平均句长放在 6138 道题之前。去掉标题不能解决全部细粒度判断问题。
3. **指定证据命中不等于答案充分性。** `1809.03391-q04` 的新第一名明确写出 IDN Tagged Corpus；`1909.11980-q01` 的新第一名明确写出 French，仍因未覆盖指定段落而记为未命中。反过来，206 道原命中中 63 道仅匹配长证据的一部分；`1908.11046-q03` 的原块没有基线模型名却命中，新块写出 Baseline-BiLSTM-CNN 却未命中。不能认定全部 206 道都是真实答案质量回退。

#### 已实现：召回问题与重排问题分离

共享接口增加可选参数 `retrieve(..., rerank_question=None)`：

- `question` 和对应的 `query_vector` 继续用于 Dense/BM25 召回。问题中的论文标题仍帮助从整个库中找到相关论文。
- 非空的 `rerank_question` 只用于候选打分；不改变候选池、过滤条件或评分指标。
- 候选输入仍是 `heading_path + 原文`，保留论文标题、章节名及来源，不直接删除摘要、不提高金标准文档权重，也不读取答案来构造问题。
- 未传 `rerank_question` 的搜索、问答/API 调用保持原行为；不会用正则从用户输入里猜测或删除标题。

评测新增 `--rerank-query {auto,original,retrieval}`：

| 策略 | 实际重排问题 | 用途 |
|---|---|---|
| `auto`（默认） | 带 `evidence` 字段的 Qasper 记录使用有效的 `question_original`；其他情况使用 `question` | Qasper 采用新方案，CRUD 难版保持难版问法 |
| `retrieval` | 始终使用 `question` | 精确恢复旧 D/E 的输入策略 |
| `original` | 显式使用有效的 `question_original`，缺失/空白/非字符串则回退到 `question` | 人为指定的对照实验 |

特别注意：CRUD 的 `questions_hard.jsonl` 也有 `question_original`，它是改写前的易版题。`auto` 不会把它换回去；在 CRUD 上显式用 `original` 会改变重排问题难度，需要另行标记实验。

汇总 JSON 的 `config.rerank_query_policy` 保存选择的策略；逐题 JSONL 保存 `retrieval_query`、`rerank_query`、`rerank_query_source`，便于确认实际输入。未启用重排时，后两个字段为 null。旧结果、问题文件、标注、数据库和拒答阈值均保持原样。

#### 验证与后续验收

自动化测试覆盖：Dense/BM25/Hybrid 召回继续使用原召回问题、重排改用显式原问题、候选标题与原文保留、CRUD 难题不退回易题、缺失原问题回退、无重排保持旧行为、旧策略复现、非法参数在调用模型前报错，以及 CLI 参数和结果字段贯通。

在原 Qasper 的 embedding 地址与 API key 配置下运行，不要改用笔记库的本地 Qwen 模型：

```bash
export EMBED_MODEL='text-embedding-3-small'
export EMBED_BASE_URL='https://api.quickrouter.ai/v1'
export RERANK_MODEL='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
export RERANK_REVISION='1427fd652930e4ba29e8149678df786c240d8825'

# 新方案：召回补标题，重排用原问题
.venv/bin/python -m evaluation.run_retrieval \
  --data evaluation/data/qasper --questions questions_ctx.jsonl \
  --reuse-index --mode dense --candidate-k 50 --k-max 10 \
  --rerank local --rerank-query auto --label 'qasper_D2_original_query'

# 需要复现旧输入时：相同参数，只把策略改回 retrieval
.venv/bin/python -m evaluation.run_retrieval \
  --data evaluation/data/qasper --questions questions_ctx.jsonl \
  --reuse-index --mode dense --candidate-k 50 --k-max 10 \
  --rerank local --rerank-query retrieval --label 'qasper_D1_legacy_query'
```

必须在完整 1451 道题、1309 道可计分题上与旧 D 对比，包括原本正确的题目；固定候选池 50、`k_max=10`、embedding 与模型 revision，检查 Hit@1、Recall@5、MRR、doc_hit 和逐题回退。不能只用 206 道已失败题证明全量收益。复用索引仍需要生成问题向量，云端 embedding 可能计费。

本次落实的是输入分离及可复现对照机制。完整数据上的收益尚未重新测定；标注补全、答案充分性评测、模型更换和性能调优仍是独立任务。后续若补标，应冻结新版本并对所有方案重算，保留原表，不能为了提高某个模型的分数单方面放宽标准。

本次实现验证：174 项自动化测试、2 个子测试及 Ruff 检查通过。另用真实本地模型，通过新 `evaluate → retrieve → LocalReranker` 流程复查 `1906.08871-q01` 与 `1903.02930-q02`：在各自固定的两个历史候选中，旧策略未命中第一名，新策略命中第一名。该检查使用受控召回与 embedding 替身，证明输入路由与真实模型接通，不是完整 Top-50 或全量质量评测。

### 11.6 本地重排模型配置冲突与启动检查（2026-11-02）

D2 运行曾在 worker 初始化阶段报 `RepositoryNotFoundError`。同一终端确认 `RERANK_MODEL=qwen3-rerank`，该值覆盖了默认 CrossEncoder；`--rerank local` 把此云端 API 别名当 Hugging Face 仓库名加载。它与 `--rerank-query auto` 的输入分离无关；前面的 gRPC `too_many_pings` 与 fork 提示也不是这次 RepositoryNotFoundError 的直接原因。

本地模式需配置可加载的 Hugging Face 仓库 ID 或本地目录；HTTP 模式的模型别名由服务商解释。切换后端时显式切换模型配置，不静默忽略用户指定的模型或退回其他模型。已验证的本地配置为：

```bash
export RERANK_MODEL='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
export RERANK_REVISION='1427fd652930e4ba29e8149678df786c240d8825'
export RERANK_CACHE_DIR='/Users/bia/PycharmProjects/PythonProject5/.rag_cache/models'
```

已补充 `LocalReranker.warmup()`，评测 CLI 在打开 Milvus 和调用 query embedding **之前**加载本地模型，并打印实际模型名与版本；配置错误会立即退出。成功加载的子进程用于后续评分，不重复启动。初始化失败提示包含模型、版本、缓存目录及本地/HTTP 配置差别，不透传上游异常正文或密钥。

新结果的 `rerank_model_load_ms` 单独记录预加载耗时，`retrieval_latency_ms.local_model_preloaded` 表明本地模型是否提前加载。CLI 中本地 Dense 检索延迟不再含模型冷启动；BM25/Hybrid 仍可能包含首次构建 BM25 索引的耗时。`duration_s` 仍包含预加载；历史 D/E 的加载耗时计入首题，比较最大延迟时须注意这一口径变化。直接调用 `evaluate` 的调用方仍自行决定是否预加载。

自动化回归验证：错误模型在生成问题向量/打开数据库前失败；预加载后复用同一进程；加载错误显示实际配置并清理子进程。原有超时、崩溃后重启与默认懒加载测试继续保留。

### 11.7 DeepSeek LLM 重排实验（2026-11-02）

68 道独立问题试跑中，Dense + 本地重排的 Hit@1 从 0.2647 提升到 0.3382，但逐题检查仍发现：摘要/结论只提到研究目标，却排在直接回答问题的实验/方法段落前。另有正确答案未被旧证据标注覆盖的“假退步”。因此本轮增加一个可比较的 LLM 重排后端，不把所有问题归因于参数规模，也不修改原标注来提高分数。

**实现与配置。** `--rerank deepseek` / `RERANK_BACKEND=deepseek` 显式启用 `DeepSeekReranker`。复用 `llm.DeepSeekClient`，默认读取项目 `.env.ocr` 的 `DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`；可用 `RERANK_DEEPSEEK_ENV` 指定其他配置文件。不继承本地/HTTP 的 `RERANK_MODEL`、`RERANK_REVISION`，不修改 `.env.ocr`。本次实际配置及成功调用的模型为 `deepseek-v4-flash-vision-exp`，使用纯文本输入。它通过 `/chat/completions` 排序，不假设聊天模型有 `/rerank` 端点。[JSON 输出接口说明](https://api-docs.deepseek.com/guides/json_mode/)

每题一次 listwise 请求，输入完整 query 与当前候选的 `heading_path + text`，要求优先排序能直接回答所问对象、数值口径、数据划分或方法步骤的证据。候选文本作为不可信数据，不执行其中的指令；不向模型提供 qid、gold_docs、参考答案或标注 evidence 字段。使用 JSON 模式、temperature=0、关闭 thinking。返回值必须是全部候选 ID 的完整排列；重复、缺漏、越界、非整数、额外字段和非法 JSON 均报错，不静默回退、不自动补齐。网络暂时失败最多重试一次；格式失败直接报错。最多 100 个候选、完整提示词最多 160000 字符，超过直接报错，不静默截断，也不拆批后直接比较不同批次的分数。

**分数与可追溯性。** 为复用现有排序接口，第一名记 0，后续为 -1、-2……；`score_type=rerank_rank`。这只是名次，不是相关性或置信度；禁用余弦门槛，并明确拒绝 `--min-rerank-score`。不能根据第一名恒为 0 判断能否回答。问答阶段仍须检查证据充分性。结果元信息记录实际模型、prompt 版本及哈希、候选与字符限制、超时、调用数和 token 用量，不输出密钥或上游错误正文。

**真实诊断结果，非完整基准。** 冻结历史 SC68 Dense 的前 10 候选，选择此前 7 道 Hit@1 退步题，再取其余题按 qid 排序的前 3 道。相同候选、相同问题分别用本地模型与 DeepSeek 排序；不调用 embedding、不打开 Milvus、不重建索引。该样本偏向已知失败题，且候选池为 10，不可与旧 D/E 的 Top-50 或全部 68/1309 题直接比较。

| 这 10 题 / 固定前 10 候选 | Dense 原序 | 本地重新排序 | DeepSeek 排序 |
|---|---:|---:|---:|
| Hit@1 | 0.7000 | 0.2000 | 0.8000 |
| Hit@5 | 0.8000 | 0.8000 | 1.0000 |
| Recall@5 | 0.8000 | 0.8000 | 1.0000 |
| MRR@10 | 0.7725 | 0.4778 | 0.8750 |
| 重排 P50（ms） | — | 163.7 | 780.3 |

10 次 DeepSeek 请求成功，输入 17781 tokens、输出 330 tokens。延迟只含该次重排（DeepSeek 含网络），本地模型提前加载；不含召回、embedding、最终回答生成。质量指标仍使用旧 evidence 匹配，不能证明答案质量或全量泛化收益。冻结输入与逐题结果分别保存在本轮工作区 `outputs/deepseek-rerank/diagnostic_inputs.json`、`diagnostic_results.json`。

后续正式对照仍使用同一份 68 题、相同的索引与候选池 50，召回和重排都使用新问题。保留本地实验结果，只新增 DeepSeek 实验：

```bash
cd /Users/bia/PycharmProjects/PythonProject5
EMBED_MODEL='text-embedding-3-small' \
EMBED_BASE_URL='https://api.quickrouter.ai/v1' \
.venv/bin/python -m evaluation.run_retrieval \
  --data evaluation/data/qasper \
  --questions /Users/bia/Documents/Codex/2026-11-02/wo/outputs/qasper-pilot100/questions_ready_68.jsonl \
  --reuse-index --mode dense --candidate-k 50 --k-max 10 \
  --rerank deepseek --rerank-query retrieval --label SC68_F_deepseek_v1
```

`EMBED_API_KEY` 沿用已有可用配置；DeepSeek key 由 `.env.ocr` 读取，不需要在命令行粘贴。全量命令尚未执行，会产生 embedding 与 DeepSeek 请求费用。初版不缓存 LLM 排序，每次运行重新请求；固定 prompt/候选仍不保证模型输出完全确定。

离线测试覆盖：50 候选单次请求及编号映射；非法/缺失/重复排名；超限输入不发请求；DeepSeek 配置与旧本地模型变量隔离；错误正文脱敏；用量记录；名次不套用拒答门槛；CLI 和实际 evaluate→retrieve→rerank_hits 路由；gold/旧问题不进入请求。测试使用假的 HTTP 响应，真实云调用单独计入上述诊断结果。

### 11.8 DeepSeek 50 候选排名格式错误的容错（2026-11-02）

用户正式运行 SC68、候选池 50 时收到 `DeepSeek ranking 必须包含全部候选的唯一整数编号`。该异常发生在响应 JSON 已解析、但完整排列校验未通过时；缺失、重复、越界、非整数编号或额外字段都可能触发。旧代码没有保留该次响应及失败 qid，不能断言当时具体属于哪种错误。此前真实验证只覆盖 10 候选，50 候选仅有模拟协议测试，验证范围不足。

修复保持原来的全池 listwise 排序方式及完整候选集合，不在本地删除重复项、补尾部、改成局部 Top-k 或回退 Dense。提示词升级为 `evidence-listwise-v2`，明确三项 JSON 示例只示意格式，实际请求携带 `expected_count` 与 `required_ids`，要求包括不相关段落在内的完整排列。

校验器返回有界诊断：期望/实际条目数、缺失/重复/越界编号及数量、非整数数量。JSON 或排名结构无效时，将安全诊断作为 `format_feedback` 连同**同一问题和完整候选**重新请求模型；默认最多两次格式重试（总共三次格式尝试）。网络层仍最多重试一次，所以最坏可产生六次 HTTP 尝试；认证/不可用等调用失败不作为格式问题重试。原始模型正文不进入日志或反馈；只记录固定错误类型与数字编号。若带反馈后的提示词超限，直接报错而不截断。

最终仍不合法时，继续显式失败，并由评测器附上 `qid`；不把查询或证据原文放进错误信息。`describe()` 增加 `format_retries`、`validation_failures`、`format_retry_calls`，便于记录重试开销。原有总 token 用量包括成功收到的无效响应及格式重试。正常一次成功的请求不增加调用。

验证分三层，不能混作全量检索结果：

1. 单元测试覆盖 50 候选缺号、重复、一基编号、字符串编号、非法 JSON、连续两次失败后成功，以及达到上限时的安全诊断；断言修复请求的候选和问题完全不变，排序必须取自模型重新返回的完整排列。
2. 真实 API 对 10 道题、每题 50 个不同候选进行验证，修复版 10/10 返回合法排列。候选由历史 Dense 前 10 加 BM25 补至 50 构成，仅验证大候选池协议稳定性；不是用户失败那次 Dense 前 50 的重现，也不是质量基准。相同输入下旧版也 10/10 成功，因此没有复现用户的具体响应错误。
3. 受控故障注入：第一轮模拟响应漏掉编号 49；后续纠错请求真实发送到 DeepSeek。模型返回完整 50 个编号，校验通过。该缺号是测试注入，不能声称服务商在此次验证中真实漏号。

检查记录保存在本轮工作区 `outputs/deepseek-ranking-fix/`，包括冻结候选、旧/新版真实调用的结构诊断及故障注入记录。后续 SC68 完整实验建议使用新标签 `SC68_F_deepseek_v2`；仍保持 `--candidate-k 50 --rerank deepseek --rerank-query retrieval`。本修复没有增加断点续跑：整轮评测失败后重跑仍会重新向量化、召回和排序，可能再次计费。

### 11.9 QuickRouter Qwen3 云端重排对照（2026-11-02）

用户确认 Qwen3 与 embedding 共用 QuickRouter 服务地址和 API key。复用现有 `HTTPReranker` 和 `--rerank http`，调用 `POST https://api.quickrouter.ai/v1/rerank`，模型名为 `qwen3-rerank`。不走本地 CrossEncoder，也不改变向量模型或索引。

对照服务商文档 https://doc.quickrouter.ai/docs/rerank.html，现有请求缺少文档列出的 `instruct`。本次为该模型默认添加官方英文指令，要求检索能够回答问题的段落；支持构造参数和 `RERANK_INSTRUCT` 覆盖，并将实际指令记录到 `describe()`/评测配置中。其他 HTTP 模型在没有显式指令时保持原请求字段。请求增加 `Accept: application/json`。保留全部候选唯一 index 和有限分数校验，不补齐缺失结果、不回退 Dense。

运行时显式将 `EMBED_API_KEY` 赋给 `RERANK_API_KEY`，不在代码里自动从其他服务继承密钥。embedding 仍使用 `text-embedding-3-small`；仅重排模型改为 `qwen3-rerank`。固定 SC68 题集、索引、dense、candidate-k=50、k-max=10、rerank-query=retrieval，使用独立标签 `SC68_G_qwen3_rerank_v1`。不设置 `--within-doc`，在全部论文中召回。比较 Hit@1、Recall@5、MRR@10 和检索延迟；云端延迟包括网络，不代表模型纯推理速度。68 题已用于调试，后续须在独立测试集复核。

验证边界：离线协议测试覆盖 50 个返回项的逆序映射、默认/自定义指令、其他模型不增加指令、非法分数/缺项/重复项及 HTTP 重试。本次执行环境没有继承用户终端的 embedding key，因此没有真实 Qwen 调用或质量/延迟结果；不能把离线测试当作云端验证。模型云端别名不能用于推断具体参数量。
