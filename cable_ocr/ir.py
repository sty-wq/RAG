"""Small dictionaries with immutable evidence references and explicit decisions."""
import copy
from .reconcile import ordered_text
from .cable import normalize

FIELDS=['序号','名称','规格型号','单位','数量']


def build_ir(meta,structure,assignments,mapping,cloud=None):
    tables=copy.deepcopy(structure['tables'])
    evidence=cloud['raw_response_ref'] if cloud else 'native_evidence.json'
    provider=cloud['provider'] if cloud else 'native_pdf'
    for table in tables:
        for row in table['rows']:
            for cell in row['cells']:
                parts=assignments.get(cell['id'],[])
                if not parts:
                    cell.update(status='UNRESOLVED',decision={'status':'UNRESOLVED','candidate_id':None,'selected_form':None})
                    continue
                raw,refs,separators=ordered_text(parts)
                candidate={'id':cell['id']+'-candidate-1','raw_text':raw,'provider':provider,
                    'profile':cloud['profile'] if cloud else None,'confidence':[p.get('confidence') for p in parts],
                    'fragments':parts,'evidence_ref':evidence,'fragment_ids':refs,
                    'assembly':{'rule':'geometry_lines_and_gaps_v1','separators':separators},
                    'request_key':cloud['request_key'] if cloud else None,
                    'normalization':None,'risk_reasons':[]}
                cell['candidates']=[candidate]
        first=table['rows'][0]['cells']
        headers=[''.join(c['candidates'][0]['raw_text'].split()) if c['candidates'] else None for c in first]
        schema_ok=headers==FIELDS
        table['headers']=headers
        table['schema_status']='VERIFIED_HEADER_GEOMETRY' if schema_ok else 'HEADER_REVIEW_REQUIRED'
        for row in table['rows']:
            for cell in row['cells']:
                if not cell['candidates']: continue
                cand=cell['candidates'][0]
                field=FIELDS[cell['column']] if schema_ok and row['role']=='data' else None
                norm=normalize(cand['raw_text'],field)
                cand['normalization']=norm
                if norm['parser']:
                    cand['risk_reasons']+=norm['parser']['reasons']
                    if norm['parser']['status']!='VALID': cand['risk_reasons'].append('PARSER_NOT_VALID')
                for part in cand['fragments']:
                    conf=part.get('confidence')
                    if isinstance(conf,dict) and isinstance(conf.get('min'),(int,float)) and conf['min']<.6:
                        cand['risk_reasons'].append('LOW_CHAR_CONFIDENCE_POC_HINT')
                if not schema_ok: cand['risk_reasons'].append('HEADER_REVIEW_REQUIRED')
                cand['risk_reasons']=sorted(set(cand['risk_reasons']))
                # PoC projections are provisional, not claimed production acceptance.
                cell.update(status='NEEDS_REVIEW',decision={'id':cell['id']+'-decision-1',
                    'status':'PROVISIONAL','candidate_id':cand['id'],
                    'selected_form':'normalized' if norm['status']=='safe_formatting' else 'raw',
                    'reason_codes':['UNVALIDATED_PHASE01_POLICY']+cand['risk_reasons'],'actor':'phase01_pipeline'})
    return {'schema_version':'phase01-v1','document_id':meta['input_hash'],'revision':1,
        'status':'NEEDS_REVIEW','scope':'STRUCTURED_TABLE_ONLY','pages':[{'canonical':meta,'tables':tables,
        'structure_gate_pre':structure['status'],'structure_issues':structure['issues'],
        'structure_gate_post':mapping,'cloud_result_ref':evidence if cloud else None}],
        'execution':{'local_ocr_enabled':False,'cloud_used':cloud is not None,
          'mode':cloud['execution_mode'] if cloud else 'NATIVE','new_ocr_http_calls':cloud['new_ocr_http_calls'] if cloud else 0}}


def selected_text(cell,form=None):
    d=cell.get('decision') or {}
    if not d.get('candidate_id'): return None
    cand=next((c for c in cell['candidates'] if c['id']==d['candidate_id']),None)
    if cand is None: return None
    form=form or d.get('selected_form')
    return cand['normalization']['text'] if form=='normalized' and cand.get('normalization') else cand['raw_text']


def project(ir):
    rows=[]; links=[]; issues=[]
    for page in ir['pages']:
        if page['structure_gate_pre']!='PRE_PASSED':
            issues.append('STRUCTURE_REVIEW_REQUIRED');continue
        for table in page['tables']:
            if table['schema_status']!='VERIFIED_HEADER_GEOMETRY':
                issues.append('HEADER_REVIEW_REQUIRED');continue
            for row in table['rows'][1:]:
                rows.append({name:selected_text(cell) for name,cell in zip(FIELDS,row['cells'])})
                links.append({name:{'cell_id':cell['id'],'candidate_id':cell['decision'].get('candidate_id'),
                              'decision_id':cell['decision'].get('id'),'status':cell['status']}
                              for name,cell in zip(FIELDS,row['cells'])})
    return {'rows':rows,'status':'NEEDS_REVIEW' if rows else 'UNRESOLVED','provisional':True,
            'field_sources':links,'issues':issues,'ir_revision':ir['revision']}
