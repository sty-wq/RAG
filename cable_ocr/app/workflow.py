import copy
import json
import os
import time
from pathlib import Path
from ..common import PhaseError, stable_hash
from ..ir import FIELDS
from ..cable import normalize
from ..pipeline import run_page
from .store import Store, ident
from .provider import AccountedProvider,file_lock


def cache_root(s):return Path(os.environ.get('CABLE_CACHE_DIR', str(s.root/'request_cache'))).resolve()
def env_path():return Path(os.environ.get('CABLE_ENV_FILE','.env')).resolve()


def ingest_document(j,store=None):
    s=store or Store()
    with file_lock(s.root/'locks'/f'ingest-{j}.lock',False) as acquired:
        if not acquired:return
        job=s.job(j)
        if job['page_count'] is not None:return
        try:
            original=s.resolve(job['original'])
            if not original.exists():raise PhaseError('ARTIFACT_MISSING')
            if original.suffix=='.pdf':
                import pypdfium2 as pdfium
                with pdfium.PdfDocument(str(original)) as doc:count=len(doc)
            else:count=1
            if not 1<=count<=50:raise PhaseError('PAGE_COUNT_LIMIT')
            with s.tx() as c:
                c.execute("UPDATE jobs SET page_count=?,status='QUEUED' WHERE id=?",(count,j))
                for p in range(count):
                    c.execute("INSERT OR IGNORE INTO page_stages(job_id,page_index,stage,status,artifact_ref,error,updated) VALUES(?,?,'INGESTED','QUEUED',NULL,NULL,?)",(j,p,time.time()))
                    s.event(c,f'page:{j}:{p}','page_process',[j,p])
        except Exception as e:
            status='FAILED';code=e.code if isinstance(e,PhaseError) else 'INGEST_FAILED'
            if job['processing_contract']=='V3' and s.resolve(job['original']).suffix=='.pdf':
                import pypdfium2.raw as pdfium_raw
                if getattr(e,'err_code',None)==pdfium_raw.FPDF_ERR_PASSWORD:status='UNSUPPORTED';code='ENCRYPTED_PDF'
            with s.tx() as c:c.execute("UPDATE jobs SET status=?,error=? WHERE id=?",(status,code,j))


def fault(stage):
    # Opt-in development failure injection only. Marker gives external harness a deterministic kill point.
    if os.environ.get('CABLE_TEST_PAUSE_STAGE')==stage:
        marker=os.environ.get('CABLE_TEST_MARKER')
        if marker:Path(marker).write_text(stage)
        time.sleep(45)


def process_page(j,p,store=None):
    s=store or Store()
    if s.job(j)['processing_contract']=='V3':
        from .workflow_v3 import process_page as process_v3
        return process_v3(j,p,s)
    with file_lock(s.root/'locks'/f'page-{j}-{p}.lock',False) as acquired:
        if not acquired:return
        record=s.one('SELECT * FROM page_stages WHERE job_id=? AND page_index=?',(j,p))
        if not record or record['status'] not in {'QUEUED','PROCESSING'}:return
        job=s.job(j)
        attempt=ident();out=s.root/f'jobs/{j}/pages/{p}/attempt-{attempt}'
        s.artifact(str(out.relative_to(s.root))+'/attempt.json',{'attempt_id':attempt,'job_id':j,'page_index':p,'status':'STARTED','created':time.time()})
        with s.tx() as c:
            c.execute("UPDATE jobs SET status='PROCESSING' WHERE id=?",(j,))
            c.execute("UPDATE page_stages SET status='PROCESSING',updated=? WHERE job_id=? AND page_index=?",(time.time(),j,p))
        def stage(value):
            s.stage(j,p,value)
            s.artifact(str(out.relative_to(s.root))+'/attempt.json',{'attempt_id':attempt,'job_id':j,'page_index':p,'stage':value,'updated':time.time()})
            fault(value)
        try:
            provider=AccountedProvider(s,env_path(),cache_root(s),j,p)
            result=run_page(s.resolve(job['original']),out,env_path(),cache_root(s),page_index=p,
                            mode=job['mode'],cloud_enabled=bool(job['cloud_enabled']),provider=provider,stage_callback=stage)
            ir=result['ir'];failure=ir.get('failure',{}).get('code')
            structure=ir['pages'][0]
            issues=structure.get('structure_issues') or []
            reason='PRE_PASSED' if structure['structure_gate_pre']=='PRE_PASSED' else '|'.join(sorted({i.get('reason',i.get('code','UNKNOWN')) for i in issues}))
            with s.tx() as c:c.execute('INSERT OR REPLACE INTO structure_observations VALUES(?,?,?,?)',(j,p,reason,time.time()))
            status='WAITING_REVIEW';err=None
            if failure:
                err=failure
                status='UNSUPPORTED' if failure=='STRUCTURE_REVIEW_REQUIRED' else 'PROVIDER_UNAVAILABLE' if failure in {'OCR_PROVIDER_UNAVAILABLE','MISSING_CREDENTIALS','AUTH_FAILED','AUTH_REQUEST_FAILED','API_BUDGET_EXHAUSTED'} else 'FAILED'
            rel=str(out.relative_to(s.root))
            fault('BEFORE_WORKFLOW_COMMIT')
            with s.tx() as c:
                c.execute('UPDATE page_stages SET stage=?,status=?,artifact_ref=?,error=?,updated=? WHERE job_id=? AND page_index=?',
                          ('WAITING_REVIEW' if status=='WAITING_REVIEW' else 'VALIDATED',status,rel,err,time.time(),j,p))
                if status=='WAITING_REVIEW':
                    for table in ir['pages'][0]['tables']:
                        for row in table['rows']:
                            for cell in row['cells']:
                                evidence={'cell':cell,'row':row,'table_id':table['id'],'artifact_ref':rel,'canonical':ir['pages'][0]['canonical']}
                                c.execute('INSERT OR IGNORE INTO review_items(id,job_id,page_index,cell_id,evidence,target_id) VALUES(?,?,?,?,?,?)',
                                          (f'{j}-{p}-{cell["id"]}',j,p,cell['id'],json.dumps(evidence,ensure_ascii=False),cell['id']))
                s.event(c,f'finalize-page:{j}:{p}','document_finalize',[j])
        except Exception as e:
            with s.tx() as c:
                c.execute("UPDATE page_stages SET status='FAILED',error=?,updated=? WHERE job_id=? AND page_index=?",(e.code if isinstance(e,PhaseError) else type(e).__name__,time.time(),j,p))
                s.event(c,f'finalize-error:{j}:{p}','document_finalize',[j])


def review(s,rid):
    r=s.one('SELECT * FROM review_items WHERE id=?',(rid,))
    if not r:raise PhaseError('NOT_FOUND')
    r['evidence']=json.loads(r['evidence'])
    r['decisions']=s.all('SELECT * FROM decisions WHERE review_id=? ORDER BY timestamp',(rid,))
    return r


def decide(s,rid,payload):
    actions={'raw','normalized','manual','unresolved'}
    if payload.get('selected_form') not in actions:raise PhaseError('INVALID_DECISION')
    if not payload.get('actor','').strip() or not payload.get('reason','').strip():raise PhaseError('ACTOR_REASON_REQUIRED')
    if not isinstance(payload.get('expected_revision'),int) or not payload.get('request_id'):raise PhaseError('REVISION_REQUEST_ID_REQUIRED')
    if payload['selected_form']=='manual' and (not isinstance(payload.get('manual_text'),str) or len(payload['manual_text'])>2000):raise PhaseError('INVALID_MANUAL_TEXT')
    h=stable_hash({'review_id':rid,**payload})
    old=s.one('SELECT * FROM decisions WHERE request_id=?',(payload['request_id'],))
    if old:
        if old['payload_hash']!=h:raise PhaseError('IDEMPOTENCY_CONFLICT')
        return old
    r=review(s,rid);is_line=r['target_kind']=='LINE';cell=r['evidence']['line' if is_line else 'cell'];candidate=None
    if is_line and payload['selected_form']=='normalized':raise PhaseError('LINE_NORMALIZATION_FORBIDDEN')
    if payload.get('candidate_id'):
        candidate=next((c for c in cell['candidates'] if c['id']==payload['candidate_id']),None)
        if candidate is None:raise PhaseError('INVALID_CANDIDATE')
    form=payload['selected_form']
    if form in {'raw','normalized'} and not candidate:raise PhaseError('CANDIDATE_REQUIRED')
    text=payload.get('manual_text') if form=='manual' else None if form=='unresolved' else candidate['raw_text'] if form=='raw' else candidate['normalization']['text']
    field=FIELDS[cell['column']] if not is_line and r['evidence']['row']['role']=='data' and cell['column']<len(FIELDS) and (s.job(r['job_id'])['processing_contract']=='V2' or r['evidence'].get('schema_status')=='VERIFIED_HEADER_GEOMETRY') else None
    if is_line and form!='unresolved' and (not text or not text.strip()):raise PhaseError('EMPTY_LINE_DECISION')
    validation={'parser':None,'status':'NOT_APPLICABLE_LINE'} if is_line else {'parser':None,'status':'NOT_APPLICABLE_MODEL_TABLE'} if r['evidence'].get('schema_status')=='MODEL_REPORTED_UNVERIFIED' else normalize(text or '',field)
    d=ident()
    with s.tx() as c:
        # Recheck idempotency under the write lock to cover simultaneous duplicate submits.
        old=c.execute('SELECT * FROM decisions WHERE request_id=?',(payload['request_id'],)).fetchone()
        if old:
            if old['payload_hash']!=h:raise PhaseError('IDEMPOTENCY_CONFLICT')
            return dict(old)
        active=c.execute('SELECT extraction_revision FROM page_stages WHERE job_id=? AND page_index=?',(r['job_id'],r['page_index'])).fetchone()
        if active is None or active[0]!=r['extraction_revision']:raise PhaseError('STALE_EXTRACTION_TARGET')
        changed=c.execute('UPDATE review_items SET revision=revision+1,status=? WHERE id=? AND revision=?',
                          ('UNRESOLVED' if form=='unresolved' else 'RESOLVED',rid,payload['expected_revision'])).rowcount
        if not changed:raise PhaseError('REVISION_CONFLICT')
        c.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                  (d,payload['request_id'],rid,payload['actor'],time.time(),payload.get('candidate_id'),form,payload.get('manual_text'),
                   payload['reason'],payload['expected_revision'],text,json.dumps(validation,ensure_ascii=False),h))
        if not is_line and candidate and form=='manual' and candidate['raw_text']!=text:
            c.execute('INSERT INTO confusion_events VALUES(?,?,?,?)',(d,candidate['raw_text'],text,json.dumps({'cell_id':cell['id'],'field':field},ensure_ascii=False)))
        c.execute("UPDATE jobs SET status='WAITING_REVIEW',revision=revision+1 WHERE id=?",(r['job_id'],))
        s.event(c,f'finalize-decision:{d}','document_finalize',[r['job_id']])
    return s.one('SELECT * FROM decisions WHERE id=?',(d,))


def finalize_document(j,store=None):
    s=store or Store()
    if s.job(j)['processing_contract']=='V3':
        from .workflow_v3 import finalize_document as finalize_v3
        return finalize_v3(j,s)
    with file_lock(s.root/'locks'/f'finalize-{j}.lock',False) as acquired:
        if not acquired:return
        job=s.job(j);revision=job['revision']
        pages=s.all('SELECT * FROM page_stages WHERE job_id=? ORDER BY page_index',(j,))
        if job['page_count'] is None or len(pages)!=job['page_count'] or any(p['status'] in {'PROCESSING','QUEUED'} for p in pages):return
        rows=[];unresolved=[];ir_pages=[];links=[]
        reviews=s.all('SELECT * FROM review_items WHERE job_id=?',(j,))
        rmap={(r['page_index'],r['cell_id']):r for r in reviews}
        decs=s.all('SELECT d.* FROM decisions d JOIN review_items r ON d.review_id=r.id WHERE r.job_id=? ORDER BY d.expected_revision',(j,))
        latest={d['review_id']:d for d in decs}
        fingerprint=stable_hash({'pages':[(p['page_index'],p['artifact_ref'],p['error']) for p in pages],
                                 'decisions':[d['id'] for d in decs], 'revision':revision})
        if job['result_ref']:
            try:
                previous=s.read(job['result_ref'])
                artifacts_present=all(not p['artifact_ref'] or s.resolve(p['artifact_ref']+'/document_ir.json').is_file() for p in pages)
                if previous.get('workflow_fingerprint')==fingerprint and artifacts_present:return previous
            except PhaseError:pass
        try:
            for p in pages:
                if p['status'] not in {'WAITING_REVIEW','FINALIZED'}:
                    unresolved.append({'page':p['page_index'],'reason':p['error'] or p['status']});continue
                page_ir=s.read(p['artifact_ref']+'/document_ir.json')['pages'][0]
                page_ir=copy.deepcopy(page_ir)
                if page_ir['structure_gate_post'].get('unassigned_fragment_count',0):
                    unresolved.append({'page':p['page_index'],'reason':'UNASSIGNED_FRAGMENTS_BLOCK'})
                for table in page_ir['tables']:
                    text_rows=[]
                    for row in table['rows']:
                        values=[];sources=[]
                        for cell in row['cells']:
                            r=rmap.get((p['page_index'],cell['id']));d=latest.get(r['id']) if r else None
                            value=d['text'] if d else None
                            cell['decision']=dict(d) if d else {'status':'UNRESOLVED'}
                            cell['status']='CONFIRMED' if d and d['selected_form']!='unresolved' else 'UNRESOLVED'
                            if value is None or value.strip()=='':unresolved.append({'page':p['page_index'],'cell_id':cell['id'],'reason':'NO_REQUIRED_VALUE'})
                            if d:
                                parsed=json.loads(d['validation']).get('parser')
                                if parsed and parsed['status']!='VALID':unresolved.append({'page':p['page_index'],'cell_id':cell['id'],'reason':'PARSER_BLOCK','parser':parsed['status']})
                            values.append(value);sources.append({'decision_id':d['id'] if d else None,'cell_id':cell['id'],'page_index':p['page_index']})
                        text_rows.append(values)
                        if row['role']=='data':links.append(sources)
                    if text_rows and text_rows[0]==FIELDS:
                        for values in text_rows[1:]:
                            row=dict(zip(FIELDS,values));rows.append(row)
                            if not (row.get('序号') or '').isdigit() or not (row.get('数量') or '').replace('.','',1).isdigit():
                                unresolved.append({'page':p['page_index'],'reason':'ROW_NUMBER_VALIDATION_BLOCK'})
                    else:unresolved.append({'page':p['page_index'],'reason':'HEADER_BLOCK'})
                ir_pages.append(page_ir)
        except PhaseError as e:
            unresolved.append({'reason':e.code})
        opened=any(r['status']=='OPEN' for r in reviews)
        if opened:status='WAITING_REVIEW'
        elif unresolved:
            kinds={p['status'] for p in pages}
            status=next(iter(kinds)) if not rows and len(kinds)==1 and kinds<= {'FAILED','UNSUPPORTED','PROVIDER_UNAVAILABLE'} else 'PARTIAL'
        else:status='COMPLETED' if rows else 'PARTIAL'
        version=ident()
        result={'workflow_fingerprint':fingerprint,'document_id':j,'status':status,'rows':rows,'unresolved':unresolved,'field_sources':links,'revision':revision,'provisional':status!='COMPLETED'}
        ir_ref=s.artifact(f'jobs/{j}/revisions/ir-{revision}-{version}.json',{'schema_version':'phase02-v1','document_id':j,'revision':revision,'status':status,'pages':ir_pages})
        result_ref=s.artifact(f'jobs/{j}/revisions/result-{revision}-{version}.json',result)
        with s.tx() as c:
            changed=c.execute('UPDATE jobs SET status=?,ir_ref=?,result_ref=? WHERE id=? AND revision=?',(status,ir_ref,result_ref,j,revision)).rowcount
            if changed and status in {'COMPLETED','PARTIAL'}:
                c.execute("UPDATE page_stages SET stage='FINALIZED',status='FINALIZED' WHERE job_id=? AND status='WAITING_REVIEW'",(j,))
        return result if changed else None
