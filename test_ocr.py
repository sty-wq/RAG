"""示例调用：直接运行可用生成的文字 PDF 演示，不发起云识别。

PyCharm 请选择本项目 .venv/bin/python（Python 3.12），原 agent 环境是 3.10。
项目根目录运行：.venv/bin/python test_ocr.py
也可运行：.venv/bin/python test_ocr.py /本地/图片.png --provider deepseek
自行指定图片或扫描 PDF 时，LIVE 模式可能产生服务商调用费用。
识别配置位于 .env.ocr，默认通道位于 ocr_config.json，结果在 .ocr_runtime。
"""
import argparse
import json
from pathlib import Path

from ocr_service import recognize_file, OCRRecognitionError, PROJECT_DIR

# 在 PyCharm 直接运行时，可以把这里改为自己的文件绝对路径。
FILE_PATH = ""


def create_demo_pdf(path: Path) -> Path:
    """用标准库生成一页文字型 PDF，不依赖外部样本或报告库。"""
    stream = b"BT /F1 18 Tf 30 220 Td (PythonProject5 OCR integration demo 12345) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def main():
    parser = argparse.ArgumentParser(description="本地文件 OCR 示例")
    parser.add_argument("file_path", nargs="?", default=FILE_PATH)
    parser.add_argument("--provider", choices=["baidu", "deepseek"], default="deepseek")
    parser.add_argument("--replay", action="store_true", help="只用已有缓存，不新增云调用")
    args = parser.parse_args()
    if args.file_path:
        path = Path(args.file_path).expanduser()
    else:
        path = create_demo_pdf(PROJECT_DIR / ".ocr_runtime/examples/demo.pdf")
        print("未指定文件：使用自动生成的文字 PDF，此例通过原生文字提取，无需调用云端。")
    try:
        # 核心调用：第一个参数就是你的本地文件路径。
        result = recognize_file(path, provider=args.provider,
                                mode="REPLAY" if args.replay else "LIVE")
    except OCRRecognitionError as exc:
        print("识别错误：", exc.code)
        if exc.result is not None:
            print("已获得的部分文字：", exc.result["text"])
            print("各页错误：", json.dumps(exc.result["errors"], ensure_ascii=False))
        return 1
    print("文件：", path)
    print("状态：", result["status"])
    print("识别文字：\n", result["text"])
    for page in result["pages"]:
        print(f"第 {page['page_index'] + 1} 页表格：", page["tables"])
    print("完整 JSON：", result["result_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
