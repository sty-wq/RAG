"""Map model-reported transcription into V3 review targets without invented boxes."""
import copy
from pathlib import Path
from PIL import Image
from .common import PhaseError,digest,save_json,stable_hash
from .ingestion import ingest
from .pipeline_v3 import run_page as native_page,mode_of

def run_page(path,out,env_path,cache_dir,page_index=0,mode='REPLAY',cloud_enabled=True,provider=None,
             stage_callback=None,evidence_checkpoint=None,resume_evidence=None,extraction_revision=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);stage=stage_callback or (lambda _:None)
    image,meta,native=ingest(Path(path),page_index,out,144);stage('OBSERVED')
    if meta['source_decision']=='NATIVE_TEXT_RELIABLE' or (resume_evidence and resume_evidence.get('evidence',{}).get('source')=='NATIVE'):
        return native_page(path,out,env_path,cache_dir,page_index,mode,cloud_enabled,provider,stage_callback,evidence_checkpoint,resume_evidence,extraction_revision)
    if resume_evidence:
        envelope=copy.deepcopy(resume_evidence);evidence=envelope['evidence']
        if envelope['identity']!=stable_hash(evidence) or evidence['input_hash']!=meta['input_hash'] or evidence['canonical_hash']!=digest((out/'canonical_page.png').read_bytes()) or evidence['source']!='DEEPSEEK_VISION':raise PhaseError('EVIDENCE_IDENTITY_MISMATCH')
    else:
        cloud=None;error=None;input_path=out/'canonical_page.png'
        if meta['kind']=='image':
            with Image.open(path) as original:
                if original.getexif().get(274,1)==1:input_path=Path(path)
        if not cloud_enabled:error='OCR_PROVIDER_UNAVAILABLE'
        else:
            stage('OCR_PENDING')
            try:
                cloud=provider.recognize_page(input_path.read_bytes(),'IMAGE_FILE' if meta['kind']=='image' else 'RENDERED_IMAGE',page_index,mode)
                stage('OCR_COMPLETE')
            except PhaseError as exc:error=exc.code
        if error and native:
            # Preserve weak native evidence when the selected cloud channel fails.
            class FailedChannel:
                def recognize_page(self,*args):raise PhaseError(error)
            return native_page(path,out,env_path,cache_dir,page_index,mode,cloud_enabled,FailedChannel(),stage_callback,evidence_checkpoint,None,extraction_revision)
        evidence={'schema_version':'deepseek-page-evidence-v1','source':'DEEPSEEK_VISION','input_hash':meta['input_hash'],
                  'canonical_hash':digest((out/'canonical_page.png').read_bytes()),'provider_input_hash':digest(input_path.read_bytes()),
                  'cloud':cloud,'error':error,'coordinate_status':'COORDINATE_SPACE_UNKNOWN','extraction_revision':extraction_revision,
                  'acquisition_status':'FAILED' if error else 'COMPLETE'}
        envelope={'identity':stable_hash(evidence),'evidence':evidence}
    save_json(out/'page_evidence.json',envelope)
    if evidence_checkpoint:evidence_checkpoint(out/'page_evidence.json')
    stage('EVIDENCE_COMMITTED')
    cloud=evidence['cloud'];blocks=[];ledger=[]
    def target(atom,tid,kind):
        risks=['MODEL_TRANSCRIPTION','POSITION_UNKNOWN']+(['UNCERTAIN_CHARACTER'] if atom['uncertain'] else [])
        ref={'fragment_id':tid,'span':[0,len(atom['text'])]}
        candidate={'id':tid+'-candidate','raw_text':atom['text'],'provider':'deepseek','profile':cloud['profile'],
                   'confidence':None,'normalization':{'text':atom['text'],'status':'unchanged','parser':None,'changes':[]},
                   'fragment_refs':[ref],'fragment_ids':[tid],'fragments':[],'evidence_ref':cloud['raw_response_ref'],
                   'request_key':cloud['request_key'],'risk_reasons':risks}
        ledger.append({'atom':tid,'owner':kind,'target_id':tid})
        return {'id':tid,'text':atom['text'],'bbox':None,'source':'DEEPSEEK_VISION','candidates':[candidate],
                'fragment_refs':[ref],'geometry_status':'UNKNOWN','order_quality':'MODEL_REPORTED',
                'review_requirement':'REQUIRED' if kind=='CELL' or atom['uncertain'] else 'OPTIONAL',
                'uncertain':atom['uncertain'],'risk_reasons':risks,'decision':None,'allow_empty':kind=='CELL'}
    def add_lines(lines):
        for a in lines:
            if not a['text'].strip():continue
            line=target(a,f'ds-p{page_index}-l{a["index"]}','LINE');line['reading_order']=len(blocks);line['ordinal']=a['index']
            blocks.append({'id':line['id']+'-block','type':'lines','bbox':None,'reading_order':len(blocks),'lines':[line]})
    if cloud:
        document=cloud['document']
        if document['type']=='table':
            rows=[]
            for row in document['rows']:
                cells=[]
                for a in row['cells']:
                    cell=target(a,f'ds-p{page_index}-r{row["row_index"]}-c{a["column_index"]}','CELL')
                    cell.update(column=a['column_index'],row=row['row_index'],merged=a.get('merged',False));cells.append(cell)
                rows.append({'index':row['row_index'],'role':'unclassified','cells':cells})
            blocks.append({'id':f'ds-p{page_index}-table','type':'table','bbox':None,'reading_order':0,
                           'schema_status':'MODEL_REPORTED_UNVERIFIED','order_quality':'MODEL_REPORTED','rows':rows})
            add_lines(document.get('outside_table_lines',[]))
        else:add_lines(document['lines'])
    has_text=any(c['candidates'][0]['raw_text'].strip() for b in blocks for c in ([c for r in b['rows'] for c in r['cells']] if b['type']=='table' else b['lines']))
    page={'page_index':page_index,'canonical':meta,'blocks':blocks,'parse_mode':mode_of(blocks),
          'evidence_identity':envelope['identity'],'evidence_ref':'page_evidence.json','extraction_revision':extraction_revision,
          'grouping_version':'deepseek_model_order_v1','enhancement_status':'MODEL_REPORTED' if blocks else 'SKIPPED_NO_EVIDENCE',
          'enhancement_issues':[{'code':'MODEL_STRUCTURE_NOT_GEOMETRICALLY_VERIFIED'}] if blocks else [],
          'acquisition_status':evidence['acquisition_status'],'error':evidence['error'],
          'content_status':'TEXT_DETECTED' if has_text else 'UNAVAILABLE' if evidence['error'] else 'NO_TEXT_DETECTED',
          'coverage_basis':'MODEL_RETURNED_UNITS_ONLY',
          'coverage':{'eligible_atoms':len(ledger),'assigned_atoms':len(ledger),'unassigned_atoms':0,'duplicate_atoms':0,'ledger':ledger}}
    ir={'schema_version':'ocr-ir-v3','pages':[page]};save_json(out/'document_ir.json',ir);stage('VALIDATED')
    return {'ir':ir,'out':str(out)}
