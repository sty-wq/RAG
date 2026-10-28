"""Allowlisted user result: no internal artifact references or provider payloads."""
def public_result(result):
    pages=[]
    for p in result.get('pages',[]):
        blocks=[]
        for b in p['blocks']:
            def text(t):return {k:t.get(k) for k in ('id','text','text_status','bbox','reading_order','order_quality','review_requirement','source','uncertain','merged')}
            if b['type']=='table':
                blocks.append({'type':'table','id':b['id'],'reading_order':b['reading_order'],'schema_status':b.get('schema_status'),
                               'rows':[{'role':r['role'],'cells':[text(c) for c in r['cells']]} for r in b['rows']]})
            else:blocks.append({'type':'lines','id':b['id'],'reading_order':b['reading_order'],'lines':[text(t) for t in b['lines']]})
        pages.append({'page_index':p['page_index'],'parse_mode':p['parse_mode'],'blocks':blocks,
                      'content_status':p.get('content_status'),'has_error':bool(p.get('error'))})
    return {'schema_version':'ocr-result-v3','parse_mode':result.get('parse_mode'),'review_status':result.get('review_status','UNREVIEWED'),
            'confirmed':result.get('confirmed',False),'pages':pages,'rows':result.get('rows',[]),
            'unresolved':[{'page':u.get('page',0)+1,'message':'此页存在未解决内容，请管理员核对'} for u in result.get('unresolved',[])]}
