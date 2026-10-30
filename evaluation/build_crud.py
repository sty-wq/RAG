"""从 CRUD-RAG 抽一个小样本，做成我们自己的评测集。

CRUD-RAG（https://github.com/IAAR-Shanghai/CRUD_RAG）的问答题分三类：
  1doc：看 1 篇新闻就能回答；2docs / 3docs：要综合 2 篇 / 3 篇新闻。
每道题都带着它依据的新闻原文（news1、news2、news3 字段）和参考答案。

我们这样抽样（固定随机种子，每次抽到的都一样）：
  1. 每类抽 per_task 道题。**同一个新闻事件只用一次**：CRUD-RAG 里三类题目会围绕同一个事件出题，
     如果两道题用了同一事件的不同报道，一道题的"标准答案文档"就可能被另一道题的文档抢了排名，
     分数会失真。
  2. 题目依据的新闻原文 = 标准答案文档，写进语料。
  3. 从 8 万篇新闻里随机抽 distractors 篇作为干扰项。同一篇新闻在语料库里可能有好几份副本，
     副本也能回答问题、却会被算成"没找对"，所以和任何标准答案文档重复的新闻都不要。
  4. 再抽 unanswerable 道 1doc 题目，但**故意不把它们的原文放进语料**：资料里没有答案，
     用来测系统会不会老实拒答。

输出（都在 out 目录下，不提交到 git）：
  corpus/xxxxxxxx.txt   一篇新闻一个文件，文件名是正文的哈希
  questions.jsonl       每行一道题：qid、类型、问题、参考答案、标准答案文档
  manifest.json         抽样参数和数量，方便复现

用法（在项目根目录运行）：
  python -m evaluation.build_crud --crud-dir ~/CRUD_RAG
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_DIR / "evaluation" / "data" / "crud"

# 类型 → (split_merged.json 里的键, 这类题依据几篇新闻)
TASKS = {
    "1doc": ("questanswer_1doc", 1),
    "2docs": ("questanswer_2docs", 2),
    "3docs": ("questanswer_3docs", 3),
}


def doc_file_name(text: str) -> str:
    """用正文的哈希当文件名：同一篇新闻永远是同一个名字，重复的自然合并。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16] + ".txt"


def squash(text: str) -> str:
    """去掉所有空白，比较"是不是同一篇"时不受空格、换行影响。"""
    return "".join(text.split())


def snippets(text: str, length: int = 20) -> list[str]:
    """从正文 1/4、1/2、3/4 处各取一小段，用来判断别的新闻是不是它的副本。"""
    flat = squash(text)
    if len(flat) < length * 2:
        return [flat] if flat else []
    return [flat[int(len(flat) * p): int(len(flat) * p) + length] for p in (0.25, 0.5, 0.75)]


def news_of(item: dict, count: int) -> list[str]:
    """取出一道题依据的 news1 ~ newsN 原文。"""
    texts = [str(item.get(f"news{i}", "")).strip() for i in range(1, count + 1)]
    return [t for t in texts if t]


def usable(item: dict, count: int) -> bool:
    return (bool(str(item.get("questions", "")).strip()) and bool(str(item.get("answers", "")).strip())
            and len(news_of(item, count)) == count)


def load_corpus_lines(corpus_dir: Path) -> list[str]:
    """8 万篇新闻：每个文件里一行就是一篇。"""
    lines: list[str] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            lines.extend(line.strip() for line in path.read_text(encoding="utf-8").split("\n"))
    return [line for line in lines if len(line) >= 50]


def build(crud_dir: Path, out_dir: Path, per_task: int = 100, unanswerable: int = 50,
          distractors: int = 3000, seed: int = 42) -> dict:
    split_file = crud_dir / "data" / "crud_split" / "split_merged.json"
    data = json.loads(split_file.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    used_events: set[str] = set()
    questions: list[dict] = []
    gold_texts: dict[str, str] = {}              # 文件名 → 正文

    # ① 每类抽 per_task 道题，同一个事件只用一次
    for qtype, (key, count) in TASKS.items():
        items = [it for it in data[key] if usable(it, count)]
        rng.shuffle(items)
        picked = 0
        for item in items:
            if picked == per_task:
                break
            event_id = str(item.get("ID", ""))
            if event_id in used_events:
                continue
            used_events.add(event_id)
            texts = news_of(item, count)
            names = []
            for text in texts:
                name = doc_file_name(text)
                gold_texts[name] = text
                names.append(name)
            picked += 1
            questions.append({
                "qid": f"{qtype}-{picked:04d}", "type": qtype, "event_id": event_id,
                "question": str(item["questions"]).strip(), "answer": str(item["answers"]).strip(),
                "gold_docs": sorted(set(names)),
            })

    # ② 无答案题：再抽一些 1doc 题，但不把它们的原文放进语料
    removed_texts: list[str] = []
    items = [it for it in data["questanswer_1doc"] if usable(it, 1)]
    rng.shuffle(items)
    picked = 0
    for item in items:
        if picked == unanswerable:
            break
        event_id = str(item.get("ID", ""))
        text = news_of(item, 1)[0]
        if event_id in used_events or doc_file_name(text) in gold_texts:
            continue
        used_events.add(event_id)
        removed_texts.append(text)
        picked += 1
        questions.append({
            "qid": f"unanswerable-{picked:04d}", "type": "unanswerable", "event_id": event_id,
            "question": str(item["questions"]).strip(), "answer": str(item["answers"]).strip(),
            "gold_docs": [], "removed_doc": doc_file_name(text),
        })

    # ③ 干扰项：随机抽新闻，跳过和标准答案 / 被拿掉的原文重复的副本
    guard = [s for text in [*gold_texts.values(), *removed_texts] for s in snippets(text)]
    guard_full = {squash(text) for text in [*gold_texts.values(), *removed_texts]}
    lines = load_corpus_lines(crud_dir / "data" / "80000_docs")
    order = list(range(len(lines)))
    rng.shuffle(order)
    distractor_texts: dict[str, str] = {}
    skipped_duplicates = 0
    for index in order:
        if len(distractor_texts) == distractors:
            break
        text = lines[index]
        flat = squash(text)
        if flat in guard_full or any(s in flat for s in guard):
            skipped_duplicates += 1
            continue
        name = doc_file_name(text)
        if name not in gold_texts:
            distractor_texts[name] = text

    # ④ 写文件
    if out_dir.exists():
        shutil.rmtree(out_dir)
    corpus_out = out_dir / "corpus"
    corpus_out.mkdir(parents=True)
    for name, text in {**distractor_texts, **gold_texts}.items():
        (corpus_out / name).write_text(text + "\n", encoding="utf-8")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    counts = {qtype: sum(q["type"] == qtype for q in questions) for qtype in [*TASKS, "unanswerable"]}
    manifest = {
        "source": "CRUD-RAG data/crud_split/split_merged.json + data/80000_docs",
        "source_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest()[:16],
        "seed": seed,
        "params": {"per_task": per_task, "unanswerable": unanswerable, "distractors": distractors},
        "questions": counts,
        "corpus_docs": len(gold_texts) + len(distractor_texts),
        "gold_docs": len(gold_texts),
        "distractor_docs": len(distractor_texts),
        "skipped_duplicates": skipped_duplicates,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 CRUD-RAG 抽样，生成评测集")
    parser.add_argument("--crud-dir", default="~/CRUD_RAG", help="CRUD-RAG 仓库克隆下来的目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（会被整个覆盖）")
    parser.add_argument("--per-task", type=int, default=100, help="1doc / 2docs / 3docs 每类各抽几道")
    parser.add_argument("--unanswerable", type=int, default=50, help="无答案题的数量")
    parser.add_argument("--distractors", type=int, default=3000, help="干扰新闻的篇数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子：同样的种子每次抽到的都一样")
    args = parser.parse_args(argv)

    manifest = build(Path(args.crud_dir).expanduser(), Path(args.out), args.per_task,
                     args.unanswerable, args.distractors, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
