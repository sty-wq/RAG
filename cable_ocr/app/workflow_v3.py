"""V3 jobs reuse the existing tasks, locks, outbox and append-only Decisions."""
import json,time
from ..common import PhaseError,stable_hash
from ..pipeline_v3 import run_page,classify,mode_of
from ..ir import FIELDS
from .store import ident
from .provider import AccountedProvider,file_lock


def targets(page):
    for b in page['blocks']:
        if b['type']=='table':
            for row in b['rows']:
                for cell in row['cells']:yield 'CELL',cell,{'row':row,'table_id':b['id'],'schema_status':b['schema_status']}
        else:
            for line in b['lines']:yield 'LINE',line,{}


def process_page(j,p,s):
    from .workflow import cache_root,env_path,fault
    with file_lock(s.root/'locks'/f'page-{j}-{p}.lock',False) as acquired:
        if not acquired:return
        rec=s.one('SELECT * FROM page_stages WHERE job_id=? AND page_index=?',(j,p))
        if not rec or rec['status'] not in {'QUEUED','PROCESSING'}:return
        job=s.job(j);out=s.root/f'jobs/{j}/pages/{p}/attempt-{ident()}';rel=str(out.relative_to(s.root))
        with s.tx() as c:
            c.execute("UPDATE jobs SET status='PROCESSING' WHERE id=?",(j,))
            c.execute("UPDATE page_stages SET status='PROCESSING' WHERE job_id=? AND page_index=?",(j,p))
        def stage(value):s.stage(j,p,value);fault(value)
        def checkpoint(path):
            with s.tx() as c:c.execute('UPDATE page_stages SET evidence_ref=? WHERE job_id=? AND page_index=?',(str(path.relative_to(s.root)),j,p))
        try:
            runner=run_page;provider_class=AccountedProvider
            if job['ocr_provider']=='deepseek':
                from ..deepseek_pipeline import run_page as runner
                from .deepseek_provider import DeepSeekProvider as provider_class
            result=runner(s.resolve(job['original']),out,env_path(),cache_root(s),p,job['mode'],bool(job['cloud_enabled']),
                            provider_class(s,env_path(),cache_root(s),j,p),stage,checkpoint,
                            s.read(rec['evidence_ref']) if rec['evidence_ref'] else None,rec['extraction_revision'])
            page=result['ir']['pages'][0];error=page['error']
            status='PARTIAL' if error and page['blocks'] else classify(error) if error else 'READY'
            if status=='ADMISSION_REJECTED':
                from ..admission import log_rejection
                log_rejection(s.root,error,{'job_id':j,'page_index':p})
            fault('BEFORE_WORKFLOW_COMMIT')
            with s.tx() as c:
                c.execute('UPDATE page_stages SET status=?,artifact_ref=?,error=?,parse_mode=?,enhancement_status=?,updated=? WHERE job_id=? AND page_index=?',
                          (status,rel,error,page['parse_mode'],page['enhancement_status'],time.time(),j,p))
                c.execute('INSERT OR REPLACE INTO structure_observations VALUES(?,?,?,?)',(j,p,page['enhancement_status'],time.time()))
                for kind,target,extra in targets(page):
                    evidence={kind.lower():target,**extra,'canonical':page['canonical'],'artifact_ref':rel,'evidence_identity':page['evidence_identity']}
                    rid=f'{j}-{p}-{target["id"]}-e{rec["extraction_revision"]}'
                    c.execute('''INSERT OR IGNORE INTO review_items(id,job_id,page_index,cell_id,evidence,target_kind,target_id,extraction_revision,review_requirement)
                       VALUES(?,?,?,?,?,?,?,?,?)''',(rid,j,p,target['id'] if kind=='CELL' else None,json.dumps(evidence,ensure_ascii=False),kind,target['id'],rec['extraction_revision'],target['review_requirement']))
                s.event(c,f'finalize-page:{j}:{p}','document_finalize',[j])
        except Exception as e:
            code=e.code if isinstance(e,PhaseError) else type(e).__name__
            if classify(code)=='ADMISSION_REJECTED':
                from ..admission import log_rejection
                log_rejection(s.root,code,{'job_id':j,'page_index':p})
            with s.tx() as c:
                c.execute("UPDATE page_stages SET status=?,error=?,updated=? WHERE job_id=? AND page_index=?",(classify(code),code,time.time(),j,p))
                s.event(c,f'finalize-error:{j}:{p}','document_finalize',[j])


def finalize_document(j,s):
    with file_lock(s.root/'locks'/f'finalize-{j}.lock',False) as acquired:
        if not acquired:return
        job=s.job(j);revision=job['revision']
        pages=s.all('SELECT * FROM page_stages WHERE job_id=? ORDER BY page_index',(j,))
        if job['page_count'] is None or len(pages)!=job['page_count'] or any(p['status'] in {'QUEUED','PROCESSING'} for p in pages):return
        reviews=s.all('''SELECT r.* FROM review_items r JOIN page_stages p ON r.job_id=p.job_id AND r.page_index=p.page_index
          AND r.extraction_revision=p.extraction_revision WHERE r.job_id=?''',(j,))
        rmap={(r['page_index'],r['target_kind'],r['target_id']):r for r in reviews}
        decs=s.all('SELECT d.* FROM decisions d JOIN review_items r ON d.review_id=r.id WHERE r.job_id=? ORDER BY d.expected_revision',(j,));latest={d['review_id']:d for d in decs}
        ir_pages=[];unresolved=[];rows=[];content_hashes=[];failures=[];delivered=False;confirmed_count=0;total=0
        for record in pages:
            p=record['page_index']
            if not record['artifact_ref']:
                failures.append(record['status']);unresolved.append({'page':p,'reason':record['error'] or 'PAGE_RESULT_MISSING'})
                ir_pages.append({'page_index':p,'blocks':[],'error':record['error'],'parse_mode':None});continue
            try:
                page=s.read(record['artifact_ref']+'/document_ir.json')['pages'][0]
                content_hashes.append(stable_hash(page))
                envelope=s.read(record['evidence_ref'])
                if envelope['identity']!=stable_hash(envelope['evidence']) or envelope['identity']!=page['evidence_identity']:raise PhaseError('EVIDENCE_IDENTITY_MISMATCH')
                if page['error']:
                    failures.append(classify(page['error']));unresolved.append({'page':p,'reason':page['error']})
                if page['coverage']['unassigned_atoms'] or page['coverage']['duplicate_atoms']:unresolved.append({'page':p,'reason':'EVIDENCE_COVERAGE_FAILED'})
                for kind,target,extra in targets(page):
                    total+=1;r=rmap.get((p,kind,target['id']));d=latest.get(r['id']) if r else None
                    target['review_requirement']=r['review_requirement'] if r else target['review_requirement']
                    target['review_id']=r['id'] if r else None
                    cand=target['candidates'][0] if target['candidates'] else None
                    value=d['text'] if d else cand['raw_text'] if cand else None
                    target['text']=value;target['text_status']='unresolved' if d and d['selected_form']=='unresolved' else 'confirmed' if d else 'provisional'
                    target['decision']=d
                    if d and d['selected_form']!='unresolved':confirmed_count+=1
                    if value and value.strip():delivered=True
                    if not r:unresolved.append({'page':p,'target_id':target['id'],'reason':'REVIEW_TARGET_MISSING'})
                    if d and (d['selected_form']=='unresolved' or value is None or (not value.strip() and not target.get('allow_empty'))):unresolved.append({'page':p,'target_id':target['id'],'reason':'UNRESOLVED'})
                    if kind=='CELL':
                        if not d:continue
                        parser=json.loads(d['validation']).get('parser')
                        if parser and parser['status']!='VALID':unresolved.append({'page':p,'cell_id':target['id'],'reason':'PARSER_BLOCK'})
                for b in page['blocks']:
                    if b['type']!='table':continue
                    if b.get('schema_status')=='MODEL_REPORTED_UNVERIFIED':continue
                    values=[[c['text'] for c in r['cells']] for r in b['rows']]
                    if values[0]!=FIELDS:unresolved.append({'page':p,'reason':'HEADER_BLOCK'})
                    for vals in values[1:]:
                        row=dict(zip(FIELDS,vals));rows.append(row)
                        if not (row.get('序号') or '').isdigit() or not (row.get('数量') or '').replace('.','',1).isdigit():unresolved.append({'page':p,'reason':'ROW_NUMBER_VALIDATION_BLOCK'})
                ir_pages.append(page)
            except (PhaseError,KeyError,TypeError) as e:
                code=e.code if isinstance(e,PhaseError) else 'INVALID_PAGE_IR'
                failures.append('FAILED');unresolved.append({'page':p,'reason':code});ir_pages.append({'page_index':p,'blocks':[],'error':code,'parse_mode':None})
        opened=any(r['status']=='OPEN' and r['review_requirement']=='REQUIRED' for r in reviews)
        status='WAITING_REVIEW' if opened else 'PARTIAL' if (unresolved or failures) and delivered else next((x for x in ('FAILED','PROVIDER_UNAVAILABLE','UNSUPPORTED','ADMISSION_REJECTED') if x in failures),'PARTIAL') if unresolved or failures else 'COMPLETED'
        review_status='CONFIRMED' if total and confirmed_count==total else 'PARTIALLY_REVIEWED' if confirmed_count else 'UNREVIEWED'
        blocks=[b for p in ir_pages for b in p['blocks']];parse_mode=mode_of(blocks)
        fingerprint=stable_hash({'pages':content_hashes,'records':pages,'reviews':reviews,'decisions':decs,'revision':revision,'failures':failures,'unresolved':unresolved})
        if job['result_ref']:
            try:
                old=s.read(job['result_ref'])
                if old.get('workflow_fingerprint')==fingerprint:return old
            except PhaseError:pass
        result={'schema_version':'ocr-result-v3','document_id':j,'status':status,'parse_mode':parse_mode,'review_status':review_status,
                'ocr_provider':job['ocr_provider'],
                'confirmed':status=='COMPLETED' and review_status=='CONFIRMED','provisional':review_status!='CONFIRMED',
                'review_policy':'cell_required_line_optional_v1','pages':ir_pages,'rows':rows,'unresolved':unresolved,
                'revision':revision,'workflow_fingerprint':fingerprint}
        version=ident();ir_ref=s.artifact(f'jobs/{j}/revisions/ir-{revision}-{version}.json',{'schema_version':'ocr-ir-v3','pages':ir_pages})
        result_ref=s.artifact(f'jobs/{j}/revisions/result-{revision}-{version}.json',result)
        with s.tx() as c:
            changed=c.execute('UPDATE jobs SET status=?,parse_mode=?,ir_ref=?,result_ref=? WHERE id=? AND revision=?',(status,parse_mode,ir_ref,result_ref,j,revision)).rowcount
        return result if changed else None


def require_line(s,rid,payload):
    """Audit an administrator's opt-in requirement; shares the target revision CAS."""
    if not all(isinstance(payload.get(k),str) and payload[k].strip() for k in ('request_id','actor','reason')):raise PhaseError('ACTOR_REASON_REQUIRED')
    h=stable_hash({'review_id':rid,**payload})
    with s.tx() as c:
        old=c.execute('SELECT * FROM review_requirement_events WHERE request_id=?',(payload['request_id'],)).fetchone()
        if old:
            if old['payload_hash']!=h:raise PhaseError('IDEMPOTENCY_CONFLICT')
            return dict(old)
        r=c.execute('SELECT * FROM review_items WHERE id=?',(rid,)).fetchone()
        if not r:raise PhaseError('NOT_FOUND')
        if r['target_kind']!='LINE':raise PhaseError('LINE_REQUIRED')
        active=c.execute('SELECT extraction_revision FROM page_stages WHERE job_id=? AND page_index=?',(r['job_id'],r['page_index'])).fetchone()
        if active[0]!=r['extraction_revision']:raise PhaseError('STALE_EXTRACTION_TARGET')
        changed=c.execute("UPDATE review_items SET review_requirement='REQUIRED',revision=revision+1 WHERE id=? AND revision=?",(rid,payload.get('expected_revision'))).rowcount
        if not changed:raise PhaseError('REVISION_CONFLICT')
        event=ident();c.execute('INSERT INTO review_requirement_events VALUES(?,?,?,?,?,?,?,?)',(event,payload['request_id'],rid,payload['actor'],payload['reason'],payload['expected_revision'],time.time(),h))
        c.execute("UPDATE jobs SET status='WAITING_REVIEW',revision=revision+1 WHERE id=?",(r['job_id'],))
        s.event(c,'require-line:'+event,'document_finalize',[r['job_id']])
    return s.one('SELECT * FROM review_requirement_events WHERE id=?',(event,))
