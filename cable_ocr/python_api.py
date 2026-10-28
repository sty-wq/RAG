"""Synchronous local-file API; no HTTP server, Celery worker or broker required."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys


class OCRRecognitionError(RuntimeError):
    """Typed failure, with a partial result when page processing has completed."""

    def __init__(self, code, result=None):
        self.code = code
        self.result = result
        super().__init__(code)


def _config_path(value, variable, filename):
    configured = value if value is not None else os.environ.get(variable)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    local = Path.cwd() / filename
    development = Path(__file__).resolve().parents[2]
    if not local.exists() and (development / 'pyproject.toml').is_file():
        local = development / filename
    return local.resolve()


def recognize(
    file_path: str | os.PathLike,
    *,
    provider: str | None = None,
    env_path: str | os.PathLike | None = None,
    config_path: str | os.PathLike | None = None,
    work_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    mode: str = 'LIVE',
    cloud_enabled: bool = True,
    timeout_seconds: float | None = None,
    raise_on_error: bool = True,
) -> dict:
    """Recognize a local PDF/PNG/JPEG and return text, tables and V3 page results.

    Only file_path is required. Provider defaults to config.json (normally Baidu).
    LIVE reuses valid cache; REPLAY never issues a new cloud request. Each call
    runs synchronously in a child Python process, so it also works from threads.
    No application server, queue or GPU/model installation is needed.

    State and evidence default to .cable_ocr_python/ in the caller's directory;
    they are separate from web-service data. New API state has no cumulative call
    cap. Native PDF text can bypass either cloud provider. Results remain
    unreviewed, even when extraction completes; WAITING_REVIEW does not block.

    Raises OCRRecognitionError for invalid input, child failures or page errors.
    For page failures, exception.result retains all available output. Set
    raise_on_error=False to return those errors in result['errors'] instead.
    """
    try:
        path = Path(file_path).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        raise OCRRecognitionError('INVALID_FILE_PATH') from None
    if not path.is_file():
        raise OCRRecognitionError('INPUT_NOT_FOUND')
    if provider not in {None, 'baidu', 'deepseek'}:
        raise OCRRecognitionError('UNSUPPORTED_CLOUD_PROVIDER')
    if mode not in {'LIVE', 'REPLAY'}:
        raise OCRRecognitionError('INVALID_MODE')
    if type(cloud_enabled) is not bool or type(raise_on_error) is not bool:
        raise OCRRecognitionError('INVALID_BOOLEAN_OPTION')
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise OCRRecognitionError('INVALID_TIMEOUT')
    work = Path(work_dir or Path.cwd() / '.cable_ocr_python').expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else work / 'cache'
    settings = _config_path(config_path, 'CABLE_CONFIG_FILE', 'config.json')
    if config_path is not None and not settings.is_file():
        raise OCRRecognitionError('CONFIG_NOT_FOUND')
    request = {'file_path': str(path), 'provider': provider, 'mode': mode,
               'cloud_enabled': cloud_enabled, 'work_dir': str(work)}
    environment = os.environ.copy()
    environment.update(
        CABLE_DATA_DIR=str(work / 'data'), CABLE_CACHE_DIR=str(cache),
        CABLE_ENV_FILE=str(_config_path(env_path, 'CABLE_ENV_FILE', '.env')),
        CABLE_CONFIG_FILE=str(settings), PYTHONDONTWRITEBYTECODE='1',
    )
    # Explicit source lookup also supports editable installs and caller threads.
    source = str(Path(__file__).resolve().parent.parent)
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [source, environment.get('PYTHONPATH')]))
    for key in ('CABLE_TEST_PAUSE_STAGE', 'CABLE_TEST_MARKER'):
        environment.pop(key, None)
    try:
        process = subprocess.run(
            [sys.executable, '-m', 'cable_ocr._sync_worker'],
            input=json.dumps(request), text=True, encoding='utf-8',
            capture_output=True, env=environment, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise OCRRecognitionError('OCR_CALL_TIMEOUT') from None
    except OSError:
        raise OCRRecognitionError('OCR_PROCESS_START_FAILED') from None
    if process.returncode != 0:
        # Never expose raw stderr, credentials or upstream HTTP error bodies.
        raise OCRRecognitionError('OCR_PROCESS_FAILED')
    try:
        response = json.loads(process.stdout)
        if not isinstance(response, dict) or type(response.get('ok')) is not bool:
            raise ValueError()
        if not response['ok']:
            raise OCRRecognitionError(response['error'])
        result = response['result']
        if not isinstance(result, dict) or not isinstance(result.get('errors'), list):
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise OCRRecognitionError('OCR_PROCESS_INVALID_RESPONSE') from None
    if result['errors'] and raise_on_error:
        raise OCRRecognitionError(result['errors'][0]['code'], result)
    return result
