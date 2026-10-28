"""Geometry-only alignment and separated V3 metrics; never manufacture GT from OCR."""
from .common import intersection,box_area
from .line_grouping import valid_box
VERSION='v3-geometric-segments-v1'


def fraction(n,d):return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def distance(a,b):
    if len(a)*len(b)>16_000_000:raise ValueError('BENCHMARK_TEXT_RESOURCE_LIMIT')
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def evaluate_lines(predictions,gt_lines,order_pairs=(),whitespace='remove_unicode'):
    """Frozen geometry: >=50% prediction area inside exactly one GT; >=50% GT width for recall.

    No text-based matching. One GT may receive several x-ordered prediction segments.
    Default CER removes Unicode whitespace only, retaining punctuation and case.
    Unmatched predictions count as insertions, unmatched GT as deletions.
    """
    if whitespace not in {'remove_unicode','preserve'}:raise ValueError('INVALID_WHITESPACE_POLICY')
    clean=lambda t:''.join(t.split()) if whitespace=='remove_unicode' else t
    groups={g['id']:[] for g in gt_lines};extras=[];ambiguous=0
    for index,p in enumerate(predictions):
        matches=[g for g in gt_lines if valid_box(p.get('bbox')) and valid_box(g.get('bbox')) and intersection(p['bbox'],g['bbox'])/box_area(p['bbox'])>=.5]
        if len(matches)==1:groups[matches[0]['id']].append((index,p))
        else:extras.append(p);ambiguous+=len(matches)>1
    edits=sum(len(clean(p['text'])) for p in extras);denom=sum(len(clean(g['text'])) for g in gt_lines);matched={};split=0
    for g in gt_lines:
        ps=sorted(groups[g['id']],key=lambda x:(x[1]['bbox'][0],x[0]))
        text=''.join(p['text'] for _,p in ps);edits+=distance(clean(g['text']),clean(text))
        spans=sorted((max(g['bbox'][0],p['bbox'][0]),min(g['bbox'][2],p['bbox'][2])) for _,p in ps)
        covered=0;end=g['bbox'][0]
        for a,b in spans:covered+=max(0,b-max(a,end));end=max(end,b)
        if ps and covered/(g['bbox'][2]-g['bbox'][0])>=.5:matched[g['id']]=min(i for i,_ in ps)
        split+=len(ps)>1
    comparable=[(a,b) for a,b in order_pairs if a in matched and b in matched]
    return {'metric_version':VERSION,'whitespace_policy':whitespace,'line_cer':fraction(edits,denom),
            'line_recall':fraction(len(matched),len(gt_lines)),
            'line_order_accuracy':fraction(sum(matched[a]<matched[b] for a,b in comparable),len(comparable)),
            'oversegmented_gt_lines':split,'unmatched_prediction_segments':len(extras),'ambiguous_cross_line_segments':ambiguous,
            'extra_characters':sum(len(clean(p['text'])) for p in extras),'gt_line_count':len(gt_lines)}


def fallback_metrics(pages,gt_fallback_page_ids=None):
    """GT IDs independently label text-containing fallback pages, including provider failures."""
    eligible=[p for p in pages if p.get('eligible_atoms',0)>0 and p.get('enhancement_status') in {'NOT_APPLICABLE','REJECTED','FAILED'} and p.get('settled')]
    success=lambda p:p.get('line_count',0)>0 and p.get('unassigned_atoms')==0 and p.get('duplicate_atoms')==0 and not p.get('line_error')
    actual={p['id'] for p in pages if success(p)}
    gt=set(gt_fallback_page_ids) if gt_fallback_page_ids is not None else None
    return {'evidence_conditioned_fallback_success':fraction(sum(success(p) for p in eligible),len(eligible)),
            'end_to_end_gt_fallback_coverage':fraction(len(actual & gt),len(gt)) if gt is not None else None,
            'provider_failure_pages':sum(p.get('provider_failure',False) for p in pages),
            'note':'Evidence coverage is not visual text recall; no combined production accuracy.'}


def evaluate_mixed(predicted_tables,gt_tables,predicted_lines,gt_lines,order_pairs=()):
    """Independent GT regions; a table needs >=85% intersection/union to count."""
    matched=set();extras=0
    for table in predicted_tables:
        matches=[]
        for g in gt_tables:
            if not valid_box(table.get('bbox')) or not valid_box(g.get('bbox')):continue
            overlap=intersection(table['bbox'],g['bbox']);union=box_area(table['bbox'])+box_area(g['bbox'])-overlap
            if overlap/union>=.85:matches.append(g['id'])
        if len(matches)==1 and matches[0] not in matched:matched.add(matches[0])
        else:extras+=1
    return {'table_region_coverage':fraction(len(matched),len(gt_tables)),'extra_or_ambiguous_tables':extras,
            'lines':evaluate_lines(predicted_lines,gt_lines,order_pairs),'note':'Region coverage does not replace structured Cell/Row/Table EM.'}
