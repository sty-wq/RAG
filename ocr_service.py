"""项目内 OCR 入口：输入本地路径，返回文字、表格和分页结果。"""
from pathlib import Path

from cable_ocr import recognize, OCRRecognitionError

PROJECT_DIR = Path(__file__).resolve().parent


def recognize_file(
    file_path: str | Path,
    *,
    provider: str | None = None,
    mode: str = "LIVE",
    timeout_seconds: float | None = None,
    raise_on_error: bool = True,
) -> dict:
    """识别 PDF/PNG/JPG/JPEG；默认通道由 ocr_config.json 决定。

    路径相对调用者当前目录解析；配置和产物始终定位到本项目。
    LIVE 可以发送云请求，REPLAY 只读本项目已有缓存。
    失败抛出 OCRRecognitionError，部分结果可从 exception.result 读取。
    """
    return recognize(
        file_path,
        provider=provider,
        env_path=PROJECT_DIR / ".env.ocr",
        config_path=PROJECT_DIR / "ocr_config.json",
        work_dir=PROJECT_DIR / ".ocr_runtime",
        mode=mode,
        timeout_seconds=timeout_seconds,
        raise_on_error=raise_on_error,
    )


__all__ = ["recognize_file", "OCRRecognitionError"]
