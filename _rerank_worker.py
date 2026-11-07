"""本地重排的独立常驻进程。只用 JSON 行通信，绝不导入 FAISS / Milvus。"""
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys


def serve(input_stream, output_stream):
    model = None
    for line in input_stream:
        try:
            request = json.loads(line)
            # 第三方的加载/进度日志走 stderr，stdout 只能出现协议响应。
            with redirect_stdout(sys.stderr):
                if request["action"] == "init":
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError:
                        response = {"ok": False, "error": "DEPENDENCY_MISSING"}
                    else:
                        model = CrossEncoder(request["model"], device=request["device"],
                                             max_length=request["max_length"], cache_folder=request["cache_dir"],
                                             revision=request["revision"], trust_remote_code=False)
                        revision = getattr(model.config, "_commit_hash", None) or request["revision"]
                        if not revision:
                            ref = (Path(request["cache_dir"]) / ("models--" + request["model"].replace("/", "--"))
                                   / "refs" / "main")
                            revision = ref.read_text().strip() if ref.is_file() else None
                        response = {"ok": True, "revision": revision}
                elif request["action"] == "score" and model is not None:
                    values = model.predict([(request["question"], doc) for doc in request["documents"]],
                                            batch_size=request["batch_size"], show_progress_bar=False)
                    scores = [float(value) for value in values]
                    if len(scores) != len(request["documents"]) or not all(math.isfinite(v) for v in scores):
                        raise ValueError("invalid scores")
                    response = {"ok": True, "scores": scores}
                else:
                    response = {"ok": False, "error": "INVALID_ACTION"}
        except Exception as exc:
            # 不把输入原文或上游异常正文回传到 HTTP 客户端。
            response = {"ok": False, "error": type(exc).__name__}
        output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
        output_stream.flush()


if __name__ == "__main__":
    serve(sys.stdin, sys.stdout)
