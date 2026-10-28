"""Evidence acquisition followed by optional strict table enhancement and residual lines."""
import copy
from pathlib import Path
from .common import PhaseError,save_json,stable_hash,digest,intersection
from .ingestion import ingest
from .structure import recover
from .reconcile import reconcile
from .ir import build_ir
from .line_grouping import group_lines,spatial_order,valid_box,VERSION


def mode_of(blocks):
    kinds={b['type'] for b in blocks}
    return 'mixed' if len(kinds)>1 else 'structured_table' if kinds=={'table'} else 'lines' if kinds=={'lines'} else None


def classify(code):
    from .admission import ADMISSION_CODES
    if code in ADMISSION_CODES:return 'ADMISSION_REJECTED'
    if code in {'UNSUPPORTED_UPLOAD_TYPE','UNSUPPORTED_INPUT','ENCRYPTED_PDF','DEEPSEEK_INPUT_DIMENSION_EXCEEDED','DEEPSEEK_INPUT_SIZE_INVALID'}:return 'UNSUPPORTED'
    if code in {'DEEPSEEK_MISSING_CREDENTIALS','DEEPSEEK_CONFIGURATION_INVALID','DEEPSEEK_CACHE_MISS'}:return 'PROVIDER_UNAVAILABLE'
    if code and (code in {'OCR_PROVIDER_UNAVAILABLE','MISSING_CREDENTIALS','AUTH_FAILED','AUTH_REQUEST_FAILED','API_BUDGET_EXHAUSTED','PREVIOUS_REQUEST_OUTCOME_UNKNOWN','BILLING_STATUS_UNKNOWN','PREVIOUS_PROVIDER_FAILURE','BAIDU_API_ERROR','MALFORMED_BAIDU_RESPONSE','INVALID_OCR_RESPONSE'} or any(x in code for x in ('TIMEOUT','NETWORK','QUOTA','REQUEST_FAILED','HTTP','RATE_LIMIT'))):return 'PROVIDER_UNAVAILABLE'
    return 'FAILED'


def run_page(path,out,env_path,cache_dir,page_index=0,mode='REPLAY',cloud_enabled=True,provider=None,
             stage_callback=None,evidence_checkpoint=None,resume_evidence=None,extraction_revision=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stage=stage_callback or (lambda _:None)
    image,meta,native=ingest(Path(path),page_index,out,144);stage('OBSERVED')
    if resume_evidence:
        envelope=copy.deepcopy(resume_evidence);evidence=envelope['evidence']
        if stable_hash(evidence)!=envelope['identity'] or evidence['input_hash']!=meta['input_hash'] or evidence['canonical_hash']!=digest((out/'canonical_page.png').read_bytes()):raise PhaseError('EVIDENCE_IDENTITY_MISMATCH')
    else:
        cloud=None;error=None;source='NATIVE';items=native;representation=None;geometry=[]
        if meta['source_decision']!='NATIVE_TEXT_RELIABLE':
            if not cloud_enabled:error='OCR_PROVIDER_UNAVAILABLE'
            else:
                strategy='IMAGE_FILE' if meta['kind']=='image' else 'RENDERED_IMAGE'
                stage('OCR_PENDING')
                try:
                    from .baidu import request_payload
                    from .provider_representation import select
                    input_bytes,representation=select(path,out,meta,mode,cache_dir)
                    try:representation['provider_payload_size']=len(request_payload(input_bytes,strategy,page_index))
                    except PhaseError as e:
                        representation['provider_payload_size']=e.details.get('form_bytes');raise
                    if hasattr(provider,'set_representation'):provider.set_representation(representation)
                    cloud=provider.recognize_page(input_bytes,strategy,page_index,mode)
                    if not isinstance(cloud,dict) or not isinstance(cloud.get('items'),list):raise PhaseError('INVALID_OCR_RESPONSE')
                    items=cloud['items'];source='CLOUD_OCR';stage('OCR_COMPLETE')
                except PhaseError as e:error=e.code
        coord=cloud.get('coordinate_status','COORDINATE_SPACE_UNKNOWN') if cloud else 'CANONICAL_INPUT_PIXELS'
        if not cloud and meta.get('rotation',0)!=0:coord='COORDINATE_SPACE_UNKNOWN'
        if cloud and cloud.get('direction') not in (None,-1,0):coord='COORDINATE_SPACE_UNKNOWN'
        if any(not isinstance(i,dict) or not isinstance(i.get('text'),str) or not isinstance(i.get('id'),str) for i in items):raise PhaseError('INVALID_OCR_RESPONSE')
        if len({i['id'] for i in items})!=len(items):raise PhaseError('DUPLICATE_FRAGMENT_ID')
        if cloud and representation:
            from .provider_representation import map_geometry
            items,geometry,coord=map_geometry(items,representation,coord,cloud.get('direction'))
            if coord=='COORDINATE_SPACE_UNKNOWN':
                representation.update(coordinate_status='UNKNOWN',coordinate_transform=None,provider_to_canonical_transform=None)
        evidence={'schema_version':'page-evidence-v1','input_hash':meta['input_hash'],
                  'canonical_hash':digest((out/'canonical_page.png').read_bytes()),'source':source,'items':items,
                  'coordinate_status':coord,'acquisition_status':'PARTIAL' if error and any(i['text'].strip() for i in items) else 'FAILED' if error else 'COMPLETE',
                  'error':error,'cloud':cloud,'extraction_revision':extraction_revision,
                  'provider_representation':representation,'fragment_geometry':geometry}
        envelope={'identity':stable_hash(evidence),'evidence':evidence}
    save_json(out/'page_evidence.json',envelope)
    if evidence_checkpoint:evidence_checkpoint(out/'page_evidence.json')
    stage('EVIDENCE_COMMITTED')
    items=[i for i in evidence['items'] if i['text'].strip()];cloud=evidence['cloud'];coord=evidence['coordinate_status']
    known=coord in {'CANONICAL_INPUT_PIXELS','VERIFIED_TRANSFORM'}
    # Invalid/out-of-page positions remain raw evidence but must never be used for table mapping.
    known=known and all(valid_box(i.get('bbox')) and 0<=i['bbox'][0]<i['bbox'][2]<=meta['width'] and 0<=i['bbox'][1]<i['bbox'][3]<=meta['height'] for i in items)
    tables=[];consumed=set();assignments={};mapping={};structure={};reason=[];enhancement='SKIPPED_NO_EVIDENCE'
    if items:
        try:
            structure=recover(image);save_json(out/'structure.json',structure);stage('STRUCTURE_READY')
            reason=structure['issues']
            if structure['status']!='PRE_PASSED':
                enhancement='NOT_APPLICABLE' if not structure['tables'] and any(i.get('reason')=='no_complete_ruled_table' for i in reason) else 'REJECTED'
            elif not known:enhancement='REJECTED';reason=[{'code':'INVALID_PROVIDER_BBOX' if any(i.get('geometry_status')=='INVALID' for i in items) else 'COORDINATE_SPACE_UNKNOWN'}]
            else:
                assignments,mapping=reconcile(structure['tables'],items,coord)
                bad=[i for i in mapping['issues'] if i['code']!='OUTSIDE_TABLE']
                atom_ids=[part['id'] for parts in assignments.values() for part in parts]
                if len(atom_ids)!=len(set(atom_ids)):bad.append({'code':'DUPLICATE_ACTIVE_ATOM'})
                if bad:enhancement='REJECTED';reason=bad
                else:
                    provisional=build_ir(meta,structure,assignments,mapping,cloud)
                    proposed=provisional['pages'][0]['tables']
                    if any(t['schema_status']!='VERIFIED_HEADER_GEOMETRY' for t in proposed):
                        enhancement='REJECTED';reason=[{'code':'SCHEMA_NOT_SUPPORTED'}]
                    else:
                        tables=proposed;enhancement='ENHANCED';reason=[]
                        consumed={r['fragment_id'] for r in mapping['records'] if r['cell_ids']}
        except Exception as e:
            tables=[];consumed=set();enhancement='FAILED';reason=[{'code':type(e).__name__}]
    save_json(out/'enhancement.json',{'status':enhancement,'issues':reason,'structure':structure,'mapping':mapping})
    stage('RECONCILED')
    # Ownership ledger counts split characters as active atoms, never also their parent.
    ledger=[]
    for item in items:
        parts=[p for ps in assignments.values() for p in ps if p.get('parent_fragment_id')==item['id']] if item['id'] in consumed else []
        if parts:
            for part in parts:ledger.append({'atom':part['id'],'parent_fragment_id':item['id'],'owner':'TABLE'})
        elif item['id'] in consumed:ledger.append({'atom':item['id'],'owner':'TABLE'})
    spans={}
    for item in items:
        cursor=0
        for ch in item.get('characters',[]):
            at=item['text'].find(ch['char'],cursor)
            if at>=0:spans[(item['id'],ch['id'])]=[at,at+len(ch['char'])];cursor=at+len(ch['char'])
    for t in tables:
        t.update(type='table',order_quality='GEOMETRIC')
        for row in t['rows']:
            for cell in row['cells']:
                cell.update(decision=None,review_requirement='REQUIRED')
                for c in cell['candidates']:
                    c['fragment_refs']=[{'fragment_id':p.get('parent_fragment_id',p['id']),'atom_id':p['id'],'span':spans.get((p.get('parent_fragment_id'),p['id']),[0,len(p['text'])])} for p in c['fragments']]
                image.crop(cell['bbox']).save(_crop(out,cell['id']))
    residual=[i for i in items if i['id'] not in consumed]
    # Native space glyphs preserve original word boundaries; whitespace-only groups emit no target.
    residual += [i for i in evidence['items'] if evidence['source']=='NATIVE' and not i['text'].strip() and valid_box(i.get('bbox')) and not any(intersection(i['bbox'],t['bbox'])>0 for t in tables)]
    try:lines=group_lines(residual,envelope['identity'],evidence['source'],known)
    except Exception:
        from .line_grouping import group_lines as atomic_grouping
        lines=atomic_grouping(residual,envelope['identity'],evidence['source'],known,max_fragments=0,deadline_seconds=0)
    # Propagate invalid provider geometry without changing grouping parameters or rules.
    invalid_ids={i['id'] for i in residual if i.get('geometry_status')=='INVALID'}
    for line in lines:
        if any(r['fragment_id'] in invalid_ids for r in line['fragment_refs']):
            line['geometry_status']='INVALID'
            line['risk_reasons']=sorted(set(line['risk_reasons']+['INVALID_PROVIDER_BBOX']))
            for candidate in line['candidates']:
                candidate['risk_reasons']=sorted(set(candidate['risk_reasons']+['INVALID_PROVIDER_BBOX']))
    expected={i['id'] for i in residual if i['text'].strip()}
    actual={r['fragment_id'] for line in lines for r in line['fragment_refs']}
    if actual & expected != expected:raise PhaseError('EVIDENCE_COVERAGE_FAILED')
    by_fragment={i['id']:i for i in residual};references={}
    for line in lines:
        for ref in line['fragment_refs']:
            if ref['fragment_id'] not in by_fragment:raise PhaseError('EVIDENCE_COVERAGE_FAILED')
            references.setdefault(ref['fragment_id'],[]).append((ref['span'],line['id']))
    for iid in expected:
        text=by_fragment[iid]['text'];previous=0;covered=0
        for span,lid in sorted(references[iid]):
            a,b=span
            if not 0<=a<b<=len(text) or (a<previous and text[a:min(previous,b)].strip()):raise PhaseError('DUPLICATE_LINE_CONSUMPTION')
            segment=text[a:b];covered+=sum(not ch.isspace() for ch in segment);previous=max(previous,b)
            if segment.strip():ledger.append({'atom':iid if len(references[iid])==1 else 'span-'+stable_hash([iid,span]),'fragment_id':iid,'span':span,'owner':'LINE','target_id':lid})
        if covered!=sum(not ch.isspace() for ch in text):raise PhaseError('EVIDENCE_COVERAGE_FAILED')
    if len({entry['atom'] for entry in ledger})!=len(ledger):raise PhaseError('DUPLICATE_ACTIVE_ATOM')
    # Single-line blocks make spatial interleaving explicit, including side notes.
    blocks=tables+[{'id':line['id']+'-block','type':'lines','bbox':line['bbox'],'ordinal':line['ordinal'],'lines':[line]} for line in lines]
    blocks=spatial_order(blocks) if len(blocks)<=12000 else blocks
    for n,b in enumerate(blocks):
        b['reading_order']=n
        if b['type']=='lines':
            line=b['lines'][0];line['reading_order']=n
            # Line crops are generated on demand by the artifact API, not thousands of files per page.
    page={'page_index':page_index,'canonical':meta,'blocks':blocks,'parse_mode':mode_of(blocks),
          'evidence_identity':envelope['identity'],'evidence_ref':'page_evidence.json','extraction_revision':extraction_revision,
          'grouping_version':VERSION,'enhancement_status':enhancement,'enhancement_issues':reason,
          'acquisition_status':evidence['acquisition_status'],'error':evidence['error'],
          'content_status':'TEXT_DETECTED' if items else 'NO_TEXT_DETECTED' if not evidence['error'] else 'UNAVAILABLE',
          'coverage':{'eligible_atoms':len(ledger),'assigned_atoms':len(ledger),'unassigned_atoms':0,'duplicate_atoms':0,'ledger':ledger}}
    ir={'schema_version':'ocr-ir-v3','pages':[page]};save_json(out/'document_ir.json',ir);stage('VALIDATED')
    return {'ir':ir,'out':str(out)}


def _crop(out,tid):
    p=out/'crops';p.mkdir(exist_ok=True);return p/(tid+'.png')
