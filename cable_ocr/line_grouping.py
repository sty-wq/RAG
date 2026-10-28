"""Conservative geometric line segments. No linguistic correction or cable parsing."""
import math
import time
from .common import bbox_union, stable_hash, intersection, box_area
VERSION='conservative_lines_v1.3_cloud_rows'


def valid_box(b):
    return isinstance(b,(list,tuple)) and len(b)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in b) and b[2]>b[0] and b[3]>b[1]


def spatial_order(units):
    """Recursive whitespace cuts: bands first, then unambiguous columns; stable fallback."""
    if len(units)<2:return units
    if any(not valid_box(u.get('bbox')) for u in units):
        for u in units:u['order_quality']='UNCERTAIN'
        return sorted(units,key=lambda u:(u.get('ordinal',0),u['id']))
    def visit(us,depth=0):
        if len(us)<2 or depth>32:return sorted(us,key=lambda u:(u['bbox'][1],u['bbox'][0],u['id']))
        for axis in (1,0):
            ordered=sorted(us,key=lambda u:u['bbox'][axis]);end=ordered[0]['bbox'][axis+2]
            cuts=[]
            for i,u in enumerate(ordered[1:],1):
                gap=u['bbox'][axis]-end
                # Vertical gutters require more evidence than ordinary letter gaps.
                threshold=0 if axis==1 else .75*min(v['bbox'][3]-v['bbox'][1] for v in us)
                if gap>threshold:cuts.append((gap,i))
                end=max(end,u['bbox'][axis+2])
            if cuts:
                _,i=max(cuts)
                return visit(ordered[:i],depth+1)+visit(ordered[i:],depth+1)
        for u in us:u['order_quality']='UNCERTAIN'
        return sorted(us,key=lambda u:(u['bbox'][1],u['bbox'][0],u['id']))
    return visit(units)


def cloud_row_groups(parts,start,deadline_seconds):
    """Join repeated aligned record rows, without assuming OCR blocks are words.

    Long gaps require at least three aligned rows and both a compact and a wide
    column. Parallel prose columns do not meet this gate. No text interpretation.
    """
    from collections import Counter
    from statistics import median
    if len(parts)<6 or any(not p['bbox'] or p['granularity']!='UNKNOWN' or p.get('overlap') for p in parts):return None
    def aligned(a,b):
        ah=a[3]-a[1];bh=b[3]-b[1];mh=min(ah,bh)
        return max(ah,bh)/mh<=1.8 and (min(a[3],b[3])-max(a[1],b[1]))/mh>=.5 and abs((a[1]+a[3]-b[1]-b[3])/2)<=.35*mh
    bands=[]
    for p in sorted(parts,key=lambda p:((p['bbox'][1]+p['bbox'][3])/2,p['bbox'][0],p['ordinal'])):
        if time.monotonic()-start>deadline_seconds:return None
        if bands and aligned(bands[-1][0]['bbox'],p['bbox']):bands[-1].append(p)
        else:bands.append([p])
    for band in bands:band.sort(key=lambda p:(p['bbox'][0],p['ordinal']))
    counts=Counter(len(b) for b in bands if 2<=len(b)<=8)
    if not counts:return None
    count,frequency=max(counts.items(),key=lambda x:(x[1],x[0]))
    if frequency<3 or frequency<.6*len(bands):return None
    pool=[b for b in bands if len(b)==count]
    height=median(p['bbox'][3]-p['bbox'][1] for b in pool for p in b)
    slots=[];compact=wide=False
    for col in range(count):
        if time.monotonic()-start>deadline_seconds:return None
        boxes=[b[col]['bbox'] for b in pool];left=[b[0] for b in boxes];right=[b[2] for b in boxes]
        if max(left)-min(left)>2*height:return None
        ratios=[(b[2]-b[0])/(b[3]-b[1]) for b in boxes]
        compact=compact or max(ratios)<=3.5
        wide=wide or median(ratios)>=4
        slots.append((min(left),max(right)))
    if not compact or not wide or any(b[0]-a[1]<.5*height for a,b in zip(slots,slots[1:])):return None
    groups=[];joined=0
    for band in bands:
        if time.monotonic()-start>deadline_seconds:return None
        matches=[]
        for p in band:
            box=p['bbox'];hit=[i for i,(l,r) in enumerate(slots) if box[0]>=l-height and box[2]<=r+height]
            if len(hit)!=1:break
            matches.append(hit[0])
        if len(band)>=2 and len(matches)==len(band) and len(set(matches))==len(matches):groups.append(band);joined+=1
        else:groups.extend([p] for p in band)
    return groups if joined>=3 else None


def group_lines(items,evidence_id,source,coordinate_known=True,max_fragments=12000,deadline_seconds=2):
    start=time.monotonic();parts=[]
    for ordinal,item in enumerate(items):
        text=item['text'];offset=0
        for segment in text.splitlines(keepends=True) or [text]:
            span=[offset,offset+len(segment)];offset+=len(segment)
            if not segment.strip() and not (source=='NATIVE' and segment and '\n' not in segment and '\r' not in segment):continue
            ref={'fragment_id':item['id'],'span':span}
            box=item.get('bbox') if coordinate_known and valid_box(item.get('bbox')) and len(segment)==len(text) else None
            parts.append({'id':stable_hash(ref)[:24],'text':segment,'bbox':box,'ref':ref,'ordinal':ordinal+span[0]/(len(text)+1),
                          'confidence':item.get('confidence'),'granularity':item.get('granularity','CHARACTER' if source=='NATIVE' else 'UNKNOWN')})
    degraded=len(parts)>max_fragments;groups=[];anchor=None
    active=[]
    if not degraded:
        for p in sorted((p for p in parts if p['bbox']),key=lambda p:p['bbox'][0]):
            active=[a for a in active if a['bbox'][2]>p['bbox'][0]]
            if len(active)>64:p['overlap_check_degraded']=True
            for a in active[-64:]:
                if intersection(a['bbox'],p['bbox'])/min(box_area(a['bbox']),box_area(p['bbox']))>.9:
                    a['overlap']=True;p['overlap']=True
            active.append(p)
            if time.monotonic()-start>deadline_seconds:degraded=True;break
    for p in sorted(parts,key=lambda p:(p['bbox'][1] if p['bbox'] else float('inf'),p['bbox'][0] if p['bbox'] else p['ordinal'],p['ordinal'],p['id'])):
        if time.monotonic()-start>deadline_seconds:degraded=True
        can_join=False
        if not degraded and anchor and p['bbox'] and p['granularity'] in {'CHARACTER','WORD'}:
            a=anchor[0];prev=anchor[-1];b=p['bbox'];ab=a['bbox']
            if ab and a['granularity'] in {'CHARACTER','WORD'}:
                h=b[3]-b[1];ah=ab[3]-ab[1];mh=min(h,ah)
                overlap=min(b[3],ab[3])-max(b[1],ab[1]);gap=b[0]-prev['bbox'][2]
                can_join=(max(h,ah)/mh<=1.8 and overlap/mh>=.5 and abs((b[1]+b[3]-ab[1]-ab[3])/2)<=.35*mh and -.15*mh<=gap<=.75*mh)
        if can_join:anchor.append(p)
        else:anchor=[p];groups.append(anchor)
    if source=='CLOUD_OCR' and coordinate_known and not degraded:
        row_groups=cloud_row_groups(parts,start,deadline_seconds)
        if row_groups is not None:groups=row_groups
    lines=[]
    for ps in groups:
        if not any(p['text'].strip() for p in ps):continue
        refs=[p['ref'] for p in ps];lid='line-'+stable_hash([evidence_id,VERSION,refs])[:24]
        text='';seps=[]
        for i,p in enumerate(ps):
            sep=''
            if i and p['bbox'][0]-ps[i-1]['bbox'][2]>.35*min(p['bbox'][3]-p['bbox'][1],ps[i-1]['bbox'][3]-ps[i-1]['bbox'][1]) and not text.endswith(' ') and not p['text'].startswith(' '):sep=' '
            if i:seps.append({'before':p['ref'],'inserted':sep,'reason':'geometry_gap'})
            text+=sep+p['text']
        box=bbox_union([p['bbox'] for p in ps]) if all(p['bbox'] for p in ps) else None
        risks=[]
        if any(p.get('overlap') for p in ps):risks.append('OVERLAPPING_FRAGMENTS')
        if any(p.get('overlap_check_degraded') for p in ps):risks.append('OVERLAP_CHECK_DEGRADED')
        if box is None:risks.append('POSITION_UNKNOWN')
        if any(p['granularity']=='UNKNOWN' for p in ps):risks.append('PROVIDER_GRANULARITY_UNKNOWN')
        if degraded:risks.append('GROUPING_DEGRADED')
        if any(isinstance(p['confidence'],dict) and isinstance(p['confidence'].get('min'),(int,float)) and p['confidence']['min']<.6 for p in ps):risks.append('LOW_CHAR_CONFIDENCE_POC_HINT')
        candidate={'id':lid+'-candidate-1','raw_text':text,'normalization':None,'fragment_refs':refs,
                   'confidence':[p['confidence'] for p in ps], 'assembly':{'rule':VERSION,'separators':seps},'risk_reasons':risks}
        lines.append({'id':lid,'bbox':box,'source':source,'ordinal':min(p['ordinal'] for p in ps),
                      'order_quality':'GEOMETRIC' if box else 'UNCERTAIN','fragment_refs':refs,'candidates':[candidate],
                      'decision':None,'review_requirement':'OPTIONAL','risk_reasons':risks})
    # Do not use quadratic whitespace recursion after the resource limit is reached.
    ordered=lines if degraded else spatial_order(lines)
    for i,line in enumerate(ordered):
        line['reading_order']=i
        if degraded:line['order_quality']='UNCERTAIN'
        if line['order_quality']=='UNCERTAIN':line['risk_reasons']=sorted(set(line['risk_reasons']+['ORDER_UNCERTAIN']))
    return ordered
