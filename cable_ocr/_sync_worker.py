"""Private subprocess implementation of the Python API, reusing V3 workflows."""
import json
from pathlib import Path
import sys

from .admission import MAX_FILE_BYTES, validate
from .app.provider import file_lock
from .app.public_v3 import public_result
from .app.store import Store
from .app.workflow import ingest_document, process_page, finalize_document
from .common import PhaseError, save_json


def _store(work):
    work.mkdir(parents=True, exist_ok=True)
    with file_lock(work / '.python-api-init.lock'):
        marker = work / '.python-api-state'
        exists = (work / 'data/workflow.sqlite3').exists()
        if exists and (not marker.is_file() or marker.read_text() != 'cable-ocr-python-api-v1\n'):
            raise PhaseError('PYTHON_API_WORK_DIR_IN_USE')
        store = Store(work / 'data')
        if not exists:
            with store.tx() as connection:
                connection.execute('UPDATE budget SET max_calls=-1 WHERE id=1')
        marker.write_text('cable-ocr-python-api-v1\n')
        return store


def run(request):
    path = Path(request['file_path'])
    # Bounded read followed by Store.upload's immutable input snapshot.
    with path.open('rb') as source:
        data = source.read(MAX_FILE_BYTES + 1)
    validate(data, path.name)
    work = Path(request['work_dir'])
    store = _store(work)
    job = store.upload(data, path.name, mode=request['mode'],
                       cloud=request['cloud_enabled'], processing_contract='V3',
                       ocr_provider=request['provider'])
    jid = job['id']
    ingest_document(jid, store)
    job = store.job(jid)
    if job['page_count'] is not None:
        for page_index in range(job['page_count']):
            process_page(jid, page_index, store)
        result = finalize_document(jid, store)
        if result is None:
            raise PhaseError('OCR_FINALIZATION_FAILED')
    else:
        result = {'status': job['status'], 'pages': [], 'parse_mode': None,
                  'review_status': 'UNREVIEWED', 'confirmed': False}
    # Use the same allowlist as web downloads, excluding raw responses/secrets.
    output = public_result(result)
    records = store.all('SELECT page_index,error FROM page_stages WHERE job_id=? ORDER BY page_index', (jid,))
    errors = [{'page_index': r['page_index'], 'code': r['error']} for r in records if r['error']]
    if job.get('error'):
        errors.insert(0, {'page_index': None, 'code': job['error']})
    if result['status'] in {'FAILED', 'PARTIAL', 'UNSUPPORTED', 'PROVIDER_UNAVAILABLE', 'ADMISSION_REJECTED'} and not errors:
        errors.append({'page_index': None, 'code': 'OCR_RESULT_INCOMPLETE'})
    texts = []
    for page in output['pages']:
        page['tables'] = []
        page['lines'] = []
        chunks = []
        for block in page['blocks']:
            if block['type'] == 'table':
                rows = [[cell['text'] if cell['text'] is not None else '' for cell in row['cells']] for row in block['rows']]
                page['tables'].append(rows)
                chunks.extend('\t'.join(row) for row in rows)
            else:
                lines = [line['text'] if line['text'] is not None else '' for line in block['lines']]
                page['lines'].extend(lines)
                chunks.extend(lines)
        page['text'] = '\n'.join(chunks)
        texts.append(page['text'])
    result_path = work / 'data/jobs' / jid / 'python-result.json'
    output.update(document_id=jid, filename=path.name, status=result['status'],
                  ocr_provider=job['ocr_provider'], text='\n\n'.join(texts),
                  errors=errors, page_count=job['page_count'], result_path=str(result_path),
                  review_required=result['status'] == 'WAITING_REVIEW')
    save_json(result_path, output)
    return output


def main():
    try:
        request = json.load(sys.stdin)
        result = run(request)
        response = {'ok': True, 'result': result}
    except PhaseError as exc:
        response = {'ok': False, 'error': exc.code}
    except OSError:
        response = {'ok': False, 'error': 'OCR_LOCAL_IO_ERROR'}
    except Exception:
        response = {'ok': False, 'error': 'OCR_PROCESSING_ERROR'}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
