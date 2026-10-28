import json
import time
from pathlib import Path
from .common import PhaseError,save_json,save_bytes,digest
from .ingestion import ingest
from .structure import recover
from .baidu import BaiduCloudOCRProvider
from .reconcile import reconcile
from .ir import build_ir,project
from .debug import overlay
from .benchmark import evaluate


def run_page(path,out,env_path,cache_dir,page_index=0,strategy='RENDERED_IMAGE',mode='REPLAY',
             cloud_enabled=True,dpi=144,gt_path=None,provider=None,stage_callback=None):
    external_provider=provider is not None
    out=Path(out);path=Path(path);start=time.monotonic()
    stage_callback=stage_callback or (lambda stage: None)
    image,meta,native=ingest(path,page_index,out,dpi)
    stage_callback("OBSERVED")
    gt=json.loads(Path(gt_path).read_text()) if gt_path else None
    if gt and gt.get('input_hash')!=meta['input_hash']:raise PhaseError('GT_INPUT_HASH_MISMATCH')
    structure=recover(image);save_json(out/'structure.json',structure)
    overlay(image,structure['tables'],[],out/'structure_overlay.png')
    stage_callback('STRUCTURE_READY')
    cloud=None; failure=None
    if structure['status']!='PRE_PASSED':
        items=[];failure={'code':structure['status'],'details':structure['issues']}
    elif meta['source_decision']=='NATIVE_TEXT_RELIABLE':
        items=native;strategy='NATIVE'
    elif not cloud_enabled:
        items=[];failure={'code':'OCR_PROVIDER_UNAVAILABLE','details':{}}
    else:
        if meta['kind']=='image':
            if strategy=='PDF_DIRECT': raise PhaseError('PDF_DIRECT_REQUIRES_PDF')
            strategy='IMAGE_FILE'
        if meta['kind']=='pdf' and strategy=='IMAGE_FILE':raise PhaseError('IMAGE_FILE_REQUIRES_IMAGE')
        payload=path.read_bytes() if strategy=='PDF_DIRECT' else (out/'canonical_page.png').read_bytes()
        provider=provider or BaiduCloudOCRProvider(env_path,cache_dir)
        stage_callback("OCR_PENDING")
        try:
            cloud=provider.recognize_page(payload,strategy,page_index,mode)
            items=cloud['items']
            stage_callback('OCR_COMPLETE')
            # Same image bytes establish pixel coordinates, unless actual response says rotated.
            if cloud.get('direction') not in (None,-1,0): cloud['coordinate_status']='COORDINATE_SPACE_UNKNOWN'
            save_json(out/f'cloud_result_{strategy.lower()}.json',cloud)
            if external_provider:
                save_json(out/f'baidu_{strategy.lower()}_raw.ref.json',{'raw_response_ref':cloud['raw_response_ref'],'request_key':cloud['request_key']})
            else:
                save_bytes(out/f'baidu_{strategy.lower()}_raw.json',Path(cloud['raw_response_ref']).read_bytes())
        except PhaseError as e:
            failure={'code':e.code,'details':e.details};items=[]
    coord=cloud['coordinate_status'] if cloud else 'CANONICAL_INPUT_PIXELS'
    assignments,mapping=reconcile(structure['tables'],items,coord)
    save_json(out/f'reconciliation_{strategy.lower()}.json',mapping)
    stage_callback('RECONCILED')
    ir=build_ir(meta,structure,assignments,mapping,cloud)
    if failure: ir.update(status=failure['code'],failure=failure)
    elif coord=='COORDINATE_SPACE_UNKNOWN': ir['status']='COORDINATE_SPACE_UNKNOWN'
    projection=project(ir)
    if failure: projection.update(status=failure['code'],failure=failure)
    elif coord=='COORDINATE_SPACE_UNKNOWN': projection['status']='COORDINATE_SPACE_UNKNOWN'
    save_json(out/'document_ir.json',ir);save_json(out/'result.json',projection)
    overlay(image,[],items,out/f'ocr_overlay_{strategy.lower()}.png',coordinate_known=coord!='COORDINATE_SPACE_UNKNOWN')
    overlay(image,structure['tables'],items,out/f'reconciliation_overlay_{strategy.lower()}.png',mapping,coord!='COORDINATE_SPACE_UNKNOWN')
    for t in structure['tables']:
        for r in t['rows']:
            for c in r['cells']:
                crop_dir=out/'crops';crop_dir.mkdir(exist_ok=True)
                image.crop(c['bbox']).save(crop_dir/(c['id']+'.png'))
    metrics=evaluate(ir,projection,gt)
    metrics.update(input_strategy=strategy,execution_mode=mode,ocr_block_count=len(items),
        new_ocr_http_calls=cloud['new_ocr_http_calls'] if cloud else 0,
        pipeline_elapsed_ms=(time.monotonic()-start)*1000,
        coordinate_status=coord,api_error=failure,
        recorded_live_latency_ms=cloud['request_metadata']['recorded_live_latency_ms'] if cloud else None,
        cloud_input_bytes=cloud['request_metadata']['input_bytes'] if cloud else None,
        current_run_cloud_latency_ms=cloud['request_metadata']['recorded_live_latency_ms'] if cloud and not cloud['cache_hit'] else None,
        cache_hit=cloud['cache_hit'] if cloud else None)
    metrics['bbox_coordinate_stability']={'status':'UNKNOWN' if coord=='COORDINATE_SPACE_UNKNOWN' else 'SINGLE_PAGE_CHECK',
        'boxes_with_locations':sum(i.get('bbox') is not None for i in items),
        'boxes_within_canonical':sum(bool(i.get('bbox')) and 0<=i['bbox'][0]<i['bbox'][2]<=meta['width'] and
            0<=i['bbox'][1]<i['bbox'][3]<=meta['height'] for i in items) if coord!='COORDINATE_SPACE_UNKNOWN' else None,
        'note':'No cross-page coordinate stability claim. PDF_DIRECT has no inferred scale.'}
    save_json(out/'benchmark.json',metrics)
    save_json(out/'artifact_manifest.json',{'input_name':path.name,'input_hash':meta['input_hash'],
        'strategy':strategy,'pdf_comparison_status':'NOT_TESTED_NO_PDF_SAMPLE' if meta['kind']=='image' else
            ('NOT_APPLICABLE_RELIABLE_NATIVE' if strategy=='NATIVE' else 'SINGLE_STRATEGY_RUN'),
        'not_applicable_artifacts':['ocr_overlay_pdf_direct.png','ocr_overlay_rendered_image.png',
            'reconciliation_overlay_pdf_direct.png','reconciliation_overlay_rendered_image.png',
            'baidu_pdf_direct_raw.json','baidu_rendered_image_raw.json'] if meta['kind']=='image' else [],
        'notes':['All projection decisions are provisional; no human-review UI in Phase 0.1.']})
    stage_callback('VALIDATED')
    return {'ir':ir,'result':projection,'benchmark':metrics,'out':str(out)}


def compare_pdf(path,out,**kwargs):
    if not Path(path).read_bytes().startswith(b'%PDF-'):
        raise PhaseError('NOT_TESTED_NO_PDF_SAMPLE')
    results={}
    for strategy in ('PDF_DIRECT','RENDERED_IMAGE'):
        results[strategy]=run_page(path,Path(out)/strategy.lower(),strategy=strategy,**kwargs)['benchmark']
    result={'strategies':results,'recommendation':'PENDING_DOMAIN_DATA',
            'coordinate_note':'PDF_DIRECT is blocked from mapping until its provider coordinate transform is independently established. No hardcoded DPI conversion.'}
    save_json(Path(out)/'pdf_strategy_comparison.json',result)
    return result
