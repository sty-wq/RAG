"""集成回归：只用本地 PDF 或模拟结果，不调用服务商。"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main as documents
import ocr_service
from cable_ocr import OCRRecognitionError
from test_ocr import create_demo_pdf


class OCRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_native_pdf_without_cloud(self):
        path = create_demo_pdf(self.root / "demo.pdf")
        (self.root / "ocr_config.json").write_text('{"ocr":{"cloud":{"enabled":true,"provider":"baidu"}}}')
        (self.root / ".env.ocr").write_text("")
        with patch.object(ocr_service, "PROJECT_DIR", self.root):
            result = ocr_service.recognize_file(path, provider="deepseek", mode="REPLAY")
        self.assertIn("OCR integration demo 12345", result["text"])
        self.assertEqual(result["page_count"], 1)
        self.assertFalse(result["errors"])
        self.assertFalse(result["confirmed"])

    def test_parsers_return_document_and_keep_structure(self):
        result = {"text": "YUY 1KV * +", "ocr_provider": "deepseek", "page_count": 1,
                  "result_path": "/tmp/job/python-result.json",
                  "pages": [{"tables": [[["YUY 1KV * +", ""]]]}]}
        for parser, name in [(documents.pdf_parser, "a.pdf"), (documents.picture_parser, "b.jpeg")]:
            with self.subTest(name=name), patch.object(documents, "recognize_file", return_value=result) as recognize:
                path = self.root / name
                document = parser(path, provider="deepseek", mode="REPLAY")
                # 路径按调用方传入的原样使用（见 DESIGN.md 第 6 节：id 目前用调用时传入的路径）
                recognize.assert_called_once_with(path, provider="deepseek", mode="REPLAY")
                self.assertIsInstance(document, documents.Document)
                self.assertEqual(document.content, result["text"])
                self.assertEqual(document.id, hashlib.md5(str(path).encode()).hexdigest())
                # meta_data 只记 OCR 结果文件的路径，不放整个结果（会被复制到每个 chunk 上）
                self.assertEqual(document.meta_data["ocr_result_path"], result["result_path"])
                self.assertNotIn("ocr_result", document.meta_data)
                self.assertEqual(document.source_path, str(path))

    def test_directory_records_ocr_failure_and_continues(self):
        (self.root / "broken.png").write_bytes(b"bad")
        (self.root / "good.pdf").write_bytes(b"fixture")
        result = {"text": "good", "ocr_provider": "baidu", "page_count": 1,
                  "result_path": "/tmp/job/python-result.json"}
        def recognize(path, **kwargs):
            if path.suffix == ".png":
                raise OCRRecognitionError("INVALID_FILE")
            return result
        with patch.object(documents, "recognize_file", side_effect=recognize):
            docs, failed, empty = documents.load_directory(str(self.root))
        self.assertEqual(len(docs), 1)
        self.assertEqual(failed[str(self.root / "broken.png")], "INVALID_FILE")
        self.assertEqual(empty, [])

    def test_missing_file_is_explicit(self):
        with self.assertRaisesRegex(OCRRecognitionError, "INPUT_NOT_FOUND"):
            ocr_service.recognize_file(self.root / "missing.png", mode="REPLAY")

    def test_jpeg_is_registered(self):
        self.assertIs(documents.parser[".jpeg"], documents.picture_parser)


if __name__ == "__main__":
    unittest.main()
