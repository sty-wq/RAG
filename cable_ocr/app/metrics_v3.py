"""Admin-only read model, bounded artifact reads with an explicit sampling limit."""
from ..common import PhaseError
from ..pipeline_v3 import classify
from ..benchmark_v3 import fallback_metrics


def metrics(s):
    records=s.all("SELECT p.* FROM page_stages p JOIN jobs j ON p.job_id=j.id WHERE j.processing_contract='V3' ORDER BY p.updated DESC LIMIT 1000")
    pages=[];representations=[];unavailable=0;invalid_pages=invalid_fragments=0
    for r in records:
        try:
            evidence=s.read(r['evidence_ref'])['evidence'] if r['evidence_ref'] else None
            if not evidence:continue
            if evidence.get('provider_representation'):representations.append({'page':f'{r["job_id"]}:{r["page_index"]}',**evidence['provider_representation']})
            invalid=sum(i.get('geometry_status')=='INVALID' for i in evidence['items']);invalid_fragments+=invalid;invalid_pages+=bool(invalid)
            ir=s.read(r['artifact_ref']+'/document_ir.json')['pages'][0] if r['artifact_ref'] else {}
            coverage=ir.get('coverage',{})
            pages.append({'id':f'{r["job_id"]}:{r["page_index"]}','eligible_atoms':coverage.get('eligible_atoms',sum(bool(i['text'].strip()) for i in evidence['items'])),
                          'enhancement_status':r['enhancement_status'] or 'FAILED','settled':r['status'] not in {'QUEUED','PROCESSING'},
                          'line_count':sum(len(b.get('lines',[])) for b in ir.get('blocks',[])),
                          'unassigned_atoms':coverage.get('unassigned_atoms'), 'duplicate_atoms':coverage.get('duplicate_atoms'),
                          'line_error':not bool(ir),'provider_failure':bool(evidence.get('error')) and classify(evidence['error'])=='PROVIDER_UNAVAILABLE'})
        except (PhaseError,KeyError,TypeError):unavailable+=1
    return {'provider_representations':representations,'invalid_provider_bbox_pages':invalid_pages,'invalid_provider_bbox_fragments':invalid_fragments,**fallback_metrics(pages),'sample_limit':1000,'sample_pages':len(records),'unreadable_artifacts':unavailable,
            'review_targets':s.all('SELECT target_kind,review_requirement,status,count(*) n FROM review_items GROUP BY target_kind,review_requirement,status'),
            'parse_modes':s.all("SELECT parse_mode,count(*) n FROM jobs WHERE processing_contract='V3' GROUP BY parse_mode"),
            'enhancements':s.all('SELECT enhancement_status,count(*) n FROM page_stages WHERE enhancement_status IS NOT NULL GROUP BY enhancement_status')}
