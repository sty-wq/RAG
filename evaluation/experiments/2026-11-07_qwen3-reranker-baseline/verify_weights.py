"""Verify downloaded safetensors against their Hugging Face LFS SHA256 keys."""
import hashlib
import os
from pathlib import Path

from common import ROOT, check, write_json
from score import MODELS


def main():
    cache = Path(os.environ.get('HF_HOME', '/workspace/hf-cache')) / 'hub'
    result = {}
    for key, (model_id, revision) in MODELS.items():
        snapshot = cache / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
        files = sorted(snapshot.glob('*.safetensors'))
        check(bool(files), f'未找到权重: {key}')
        rows = []
        for file in files:
            target = file.resolve()
            expected = target.name
            check(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), '非预期的 LFS 缓存结构')
            hasher = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    hasher.update(block)
            actual = hasher.hexdigest()
            check(actual == expected, f'权重校验失败: {model_id}/{file.name}')
            rows.append({'file': file.name, 'bytes': file.stat().st_size, 'expected_sha256': expected,
                         'actual_sha256': actual, 'match': True})
            print(key, file.name, 'SHA256 verified', flush=True)
        result[key] = {'model': model_id, 'revision': revision, 'files': rows}
    write_json(ROOT / 'weight-integrity.json', result)


if __name__ == '__main__':
    main()
