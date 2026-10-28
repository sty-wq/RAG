"""Score against independent manual GT, never derive GT from OCR."""
from .common import box_area, intersection, contains_center
from .ir import FIELDS,selected_text
from .cable import normalize


def ratio(correct,total):
    return {'correct':correct,'total':total,'rate':correct/total if total else None}


def evaluate(ir,projection,gt):
    if not gt or gt.get('status') not in {'MANUALLY_TRANSCRIBED','VISUAL_GT_DRAFT','SYNTHETIC_GT','DATASET_GT','HUMAN_SIGNED_GT'}:
        return {'status':'NOT_EVALUATED_NO_MANUAL_GT'}
    tables=ir['pages'][0]['tables']; table=tables[0] if len(tables)==1 else None
    expected=[dict(zip(FIELDS,gt['headers']))]+gt['rows']
    counts={'raw_cell':0,'normalized_cell':0,'raw_cable':0,'normalized_cable':0,'row':0}
    total=len(expected)*len(FIELDS)
    for ri,row in enumerate(expected):
        row_ok=True
        for ci,name in enumerate(FIELDS):
            cell=table['rows'][ri]['cells'][ci] if table and ri<len(table['rows']) and ci<table['column_count'] else None
            raw=selected_text(cell,'raw') if cell else None
            norm=selected_text(cell,'normalized') if cell else None
            truth=row[name]
            normalized_truth=normalize(truth,name if ri else None)['text']
            raw_ok=raw==truth; norm_ok=norm==normalized_truth
            counts['raw_cell']+=raw_ok;counts['normalized_cell']+=norm_ok
            if ri and name=='规格型号':
                counts['raw_cable']+=raw_ok;counts['normalized_cable']+=norm_ok
            row_ok &= norm_ok
        if ri: counts['row']+=row_ok
    mapping=ir['pages'][0]['structure_gate_post']; records=mapping['records']
    gt_boxes=gt.get('cells',[]); id2cell={c['id']:c for t in tables for r in t['rows'] for c in r['cells']}
    mapping_correct=0; mapping_total=0
    for record in records:
        bbox=record.get('bbox')
        if not bbox or not gt_boxes: continue
        # Manual Cell rectangles define true geometric membership independently.
        owners=[c for c in gt_boxes if contains_center(c['bbox'],bbox) and intersection(c['bbox'],bbox)/box_area(bbox)>=.85]
        if not owners: continue
        mapping_total+=1
        actual=[id2cell[i] for i in record['cell_ids'] if i in id2cell]
        mapping_correct+=len(actual)==1 and actual[0]['row']==owners[0]['row'] and actual[0]['column']==owners[0]['column']
    return {'status':'DEVELOPMENT_REGRESSION_ONLY','ground_truth_source':gt['source'],
            'gt_human_signed_off':gt.get('human_signed_off',False),
            'raw_cell_exact_match':ratio(counts['raw_cell'],total),
            'normalized_cell_exact_match':ratio(counts['normalized_cell'],total),
            'raw_cable_model_exact_match':ratio(counts['raw_cable'],len(gt['rows'])),
            'cable_model_exact_match':ratio(counts['normalized_cable'],len(gt['rows'])),
            'row_exact_match':ratio(counts['row'],len(gt['rows'])),
            'mapping_accuracy':ratio(mapping_correct,mapping_total),
            'mapping_metric_note':'Fragment membership against manual GT Cell boxes; excludes intrinsically crossing fragments, reported separately.',
            'unassigned_fragment_count':mapping['unassigned_fragment_count'],
            'missing_cell_candidate_count':mapping['missing_cell_candidate_count'],
            'ambiguous_fragment_count':mapping['ambiguous_fragment_count'],
            'expected_grid':[len(expected),len(FIELDS)],
            'actual_grids':[[t['row_count'],t['column_count']] for t in tables],
            'extra_rows':max(0,sum(t['row_count']-1 for t in tables)-len(gt['rows'])),
            'production_accuracy_claim':False}
