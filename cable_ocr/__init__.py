"""Evidence-preserving OCR with switchable Baidu and DeepSeek channels."""

__version__ = "0.6.0"

from .python_api import recognize, OCRRecognitionError

__all__ = ['recognize', 'OCRRecognitionError', '__version__']
