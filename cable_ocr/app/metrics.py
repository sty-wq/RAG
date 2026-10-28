from ..common import intersection,box_area
from ..ir import FIELDS,selected_text
from ..benchmark import evaluate,ratio


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


def metrics(result,gt):
    ir=result['ir'];page=ir['pages'][0];cells=[c for t in page['tables'] for r in t['rows'] for c in r['cells']]
    mapping=page['structure_gate_post']
    out={'review_rate':ratio(len(cells),len(cells)), # explicit conservative all-cell review policy
         'unresolved_rate':ratio(sum(not c['candidates'] for c in cells),len(cells)),
         'unassigned_fragments':mapping['unassigned_fragment_count'],'missing_candidates':mapping['missing_cell_candidate_count']}
    if not gt:return {**out,'gt_source':'NO_GT','raw_cer':None,'structure_precision':None,'structure_recall':None}
    out['gt_source']=gt['status']
    boxes=gt.get('cells',[]);valid=gt.get('mapping_gt_valid',True)
    matched=set();tp=0
    for c in cells:
        options=[(intersection(c['bbox'],g['bbox'])/max(1,box_area(c['bbox'])+box_area(g['bbox'])-intersection(c['bbox'],g['bbox'])),i) for i,g in enumerate(boxes) if i not in matched]
        best=max(options,default=(0,None))
        if best[0]>=.8:matched.add(best[1]);tp+=1
    out.update(structure_precision=ratio(tp,len(cells)) if valid else None,structure_recall=ratio(tp,len(boxes)) if valid else None)
    if gt.get('rows'):
        out.update(evaluate(ir,result['result'],gt))
        errors=total=0
        lookup={(c['row'],c['column']):c for c in cells} if len(page['tables'])==1 else {}
        expected=[dict(zip(FIELDS,gt['headers']))]+gt['rows']
        for ri,row in enumerate(expected):
            for ci,f in enumerate(FIELDS):
                truth=row.get(f,'');c=lookup.get((ri,ci));raw=selected_text(c,'raw') if c else ''
                errors+=distance(raw or '',truth);total+=len(truth)
        # Extra recovered cells count as insertions.
        for pos,c in lookup.items():
            if pos[0]>=len(expected) or pos[1]>=len(FIELDS):errors+=len(selected_text(c,'raw') or '')
        out['raw_cer']={'errors':errors,'characters':total,'rate':errors/total if total else None}
        if not valid:out['mapping_accuracy']=None
    else:out['raw_cer']=None
    return out
