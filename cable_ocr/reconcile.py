"""Geometry only. Provider order and text similarity never select a Cell."""
import copy
from .common import box_area, intersection, contains_center, bbox_union


def owners(bbox,cells):
    if bbox is None or box_area(bbox)<=0: return []
    return [c for c in cells if contains_center(c['bbox'],bbox) and intersection(c['bbox'],bbox)/box_area(bbox)>=.85]


def ordered_text(parts):
    lines=[]
    for p in sorted(parts,key=lambda p:((p['bbox'][1]+p['bbox'][3])/2,p['bbox'][0])):
        cy=(p['bbox'][1]+p['bbox'][3])/2; height=p['bbox'][3]-p['bbox'][1]
        line=next((ln for ln in lines if abs(ln['cy']-cy)<=.45*min(ln['height'],height)),None)
        if line is None: lines.append({'cy':cy,'height':height,'parts':[p]})
        else: line['parts'].append(p)
    texts=[]; separators=[]; ids=[]
    for li,ln in enumerate(sorted(lines,key=lambda x:x['cy'])):
        ps=sorted(ln['parts'],key=lambda p:p['bbox'][0]); line_text=''
        for i,p in enumerate(ps):
            sep=''
            if i:
                prev=ps[i-1]
                gap=p['bbox'][0]-prev['bbox'][2]
                height=min(p['bbox'][3]-p['bbox'][1],prev['bbox'][3]-prev['bbox'][1])
                if gap>height*.35 and not line_text.endswith(' ') and not p['text'].startswith(' '): sep=' '
                separators.append({'before':p['id'],'inserted':sep,'reason':'geometry_gap'})
            line_text+=sep+p['text']; ids.append(p['id'])
        texts.append(line_text)
        if li: separators.append({'before_line':li,'inserted':'\n','reason':'geometry_line_break'})
    return '\n'.join(texts),ids,separators


def reconcile(tables,items,coordinate_status='CANONICAL_INPUT_PIXELS'):
    cells=[c for t in tables for r in t['rows'] for c in r['cells']]
    assignments={c['id']:[] for c in cells}; records=[]; issues=[]
    if coordinate_status not in {'CANONICAL_INPUT_PIXELS','VERIFIED_TRANSFORM'}:
        return assignments,{'status':'COORDINATE_SPACE_UNKNOWN','records':[],
            'issues':[{'code':'COORDINATE_SPACE_UNKNOWN'}],'unassigned_fragment_count':len(items),
            'missing_cell_candidate_count':len(cells),'ambiguous_fragment_count':0}
    seen_ids=set()
    for item in items:
        iid=item['id']; bbox=item.get('bbox')
        if iid in seen_ids:
            issues.append({'code':'DUPLICATE_FRAGMENT_ID','fragment_id':iid});continue
        seen_ids.add(iid)
        matches=owners(bbox,cells)
        if len(matches)==1:
            cid=matches[0]['id']; assignments[cid].append(copy.deepcopy(item))
            records.append({'fragment_id':iid,'cell_ids':[cid],'status':'MAPPED','method':'center_and_fragment_containment',
                            'containment':intersection(bbox,matches[0]['bbox'])/box_area(bbox),'bbox':bbox})
            continue
        chars=item.get('characters') or []
        # Require complete non-space character evidence before splitting a block.
        comparable=lambda text: ''.join(text.split())
        complete=bool(chars) and comparable(''.join(c['char'] for c in chars))==comparable(item['text'])
        grouped={}; char_ok=complete
        for ch in chars:
            if not ch['char'].strip(): continue
            cm=owners(ch.get('bbox'),cells)
            if len(cm)!=1: char_ok=False;break
            grouped.setdefault(cm[0]['id'],[]).append(ch)
        if char_ok and len(grouped)>=2:
            for cid,chrs in grouped.items():
                # Preserve each character as a geometric part. No dictionary split.
                for ch in chrs:
                    assignments[cid].append({'id':ch['id'],'parent_fragment_id':iid,'text':ch['char'],
                        'bbox':ch['bbox'],'confidence':ch.get('confidence'),'characters':[ch]})
            records.append({'fragment_id':iid,'cell_ids':list(grouped),'status':'SPLIT_BY_CHARACTER_GEOMETRY','bbox':bbox})
        else:
            touches=bool(bbox and any(intersection(bbox,t['bbox'])>0 for t in tables))
            code='AMBIGUOUS_MAPPING' if touches else ('MISSING_FRAGMENT_BBOX' if bbox is None else 'OUTSIDE_TABLE')
            issues.append({'code':code,'fragment_id':iid,'bbox':bbox})
            records.append({'fragment_id':iid,'cell_ids':[],'status':code,'bbox':bbox})
    # Overlapping duplicate detections are a risk even if IDs differ.
    for cid,parts in assignments.items():
        for i,a in enumerate(parts):
            for b in parts[i+1:]:
                if intersection(a['bbox'],b['bbox'])/min(box_area(a['bbox']),box_area(b['bbox']))>.9:
                    issues.append({'code':'OVERLAPPING_FRAGMENTS','cell_id':cid,'fragment_ids':[a['id'],b['id']]})
    missing=sum(not v for v in assignments.values())
    inside_issues=[i for i in issues if i['code']!='OUTSIDE_TABLE']
    return assignments,{'status':'NEEDS_REVIEW' if inside_issues or missing else 'POST_PASSED','records':records,'issues':issues,
        'unassigned_fragment_count':sum(not r['cell_ids'] for r in records),'missing_cell_candidate_count':missing,
        'ambiguous_fragment_count':sum(i['code']=='AMBIGUOUS_MAPPING' for i in issues)}
