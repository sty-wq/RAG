"""必须在新进程中测配置：PyMilvus 的环境变量解析发生在模块导入期间。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parent.parent


def probe(tmp_path, variables, explicit=None, dotenv=None):
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    shutil.copy2(PROJECT / "vector_store.py", module_dir / "vector_store.py")
    if dotenv is not None:
        (module_dir / ".env").write_text(dotenv, encoding="utf-8")
    environment = os.environ.copy()
    for name in ("MILVUS_URI", "MILVUS_DEFAULT_CONNECTION", "RAG_MILVUS_URI", "PYTHON_DOTENV_DISABLED"):
        environment.pop(name, None)
    environment.update(variables)
    environment["PYTHONPATH"] = os.pathsep.join((str(module_dir), str(PROJECT)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = '''import json, os, sys
from types import SimpleNamespace
import vector_store
from pymilvus.settings import Config
vector_store.MilvusClient = lambda uri: SimpleNamespace(uri=uri, close=lambda: None)
explicit = json.loads(sys.argv[1])
with vector_store.VectorStore(uri=explicit) as store:
    print(json.dumps({"uri": store.uri, "client_uri": store.client.uri,
                      "default_uri": vector_store.DEFAULT_URI,
                      "sdk_uri": Config.MILVUS_URI,
                      "project_uri": os.getenv("RAG_MILVUS_URI")}))
'''
    process = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(explicit)],
                              cwd=module_dir, env=environment, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


@pytest.mark.parametrize("variables,explicit,expected,sdk_uri", [
    ({"RAG_MILVUS_URI": "new.db"}, None, "new.db", ""),
    ({"MILVUS_URI": "旧目录 with spaces/old.db"}, None, "旧目录 with spaces/old.db", ""),
    ({"RAG_MILVUS_URI": "new.db", "MILVUS_URI": "old.db"}, None, "new.db", ""),
    ({"MILVUS_URI": "http://127.0.0.1:19530"}, None, "http://127.0.0.1:19530", "http://127.0.0.1:19530"),
    ({"MILVUS_URI": "", "RAG_MILVUS_URI": ""}, None, None, ""),
    ({"RAG_MILVUS_URI": "", "MILVUS_URI": "old.db"}, None, "old.db", ""),
    ({"RAG_MILVUS_URI": "new.db", "MILVUS_URI": "http://127.0.0.1:19530"}, "argument.db", "argument.db", "http://127.0.0.1:19530"),
])
def test_fresh_process_import_and_uri_precedence(tmp_path, variables, explicit, expected, sdk_uri):
    result = probe(tmp_path, variables, explicit)
    assert result["uri"] == (expected if expected is not None else result["default_uri"])
    assert result["client_uri"] == result["uri"]
    assert result["sdk_uri"] == sdk_uri


def test_legacy_local_path_from_dotenv_is_migrated_before_sdk_import(tmp_path):
    result = probe(tmp_path, {}, dotenv='MILVUS_URI="旧目录 with spaces/from-env.db"\n')
    assert result["uri"] == "旧目录 with spaces/from-env.db"
    assert result["project_uri"] == result["uri"]
    assert result["sdk_uri"] == ""
