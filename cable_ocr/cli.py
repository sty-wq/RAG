import argparse
import json
from pathlib import Path
from .common import PhaseError
from .pipeline import run_page,compare_pdf


def main():
    p=argparse.ArgumentParser(description='Cable OCR Phase 0.2: audited page LIVE/REPLAY; no local OCR')
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['LIVE','REPLAY','FORCE_LIVE'],default='REPLAY')
    p.add_argument('--force-reason',help='Required reason for FORCE_LIVE experiment')
    p.add_argument('--strategy',choices=['PDF_DIRECT','RENDERED_IMAGE','IMAGE_FILE','both'],default='RENDERED_IMAGE')
    p.add_argument('--page-index',type=int,default=0,help='zero-based; Baidu PDF page number is index+1')
    p.add_argument('--dpi',type=int,default=144)
    p.add_argument('--env',type=Path,default=Path('.env'))
    p.add_argument('--cache-dir',type=Path,default=Path('artifacts/request_cache'))
    p.add_argument('--gt',type=Path)
    p.add_argument('--cloud-disabled',action='store_true')
    p.add_argument('--config',type=Path,help='Optional config.json; local OCR must remain false')
    args=p.parse_args()
    kw=dict(env_path=args.env,cache_dir=args.cache_dir,page_index=args.page_index,mode=args.mode,
            cloud_enabled=not args.cloud_disabled,dpi=args.dpi,gt_path=args.gt)
    try:
        from .app.store import Store
        from .app.provider import AccountedProvider
        from .app.settings import switches
        from .common import digest
        if not args.input.is_file():raise PhaseError('INPUT_NOT_FOUND')
        kw['provider']=AccountedProvider(Store(),args.env,args.cache_dir,'cli:'+digest(args.input.read_bytes()),args.page_index,force_reason=args.force_reason)
        kw['cloud_enabled']=switches()['cloud'] and not args.cloud_disabled
        if args.config:
            try: cfg=json.loads(args.config.read_text())
            except Exception: raise PhaseError('INVALID_CONFIG') from None
            if cfg.get('ocr',{}).get('local',{}).get('enabled',False): raise PhaseError('LOCAL_OCR_NOT_IMPLEMENTED')
            cloud=cfg.get('ocr',{}).get('cloud',{})
            if cloud.get('provider','baidu')!='baidu': raise PhaseError('UNSUPPORTED_PROVIDER')
            kw['cloud_enabled']=cloud.get('enabled',True) and not args.cloud_disabled
        if args.strategy=='both': result=compare_pdf(args.input,args.output,**kw)
        else:
            r=run_page(args.input,args.output,strategy=args.strategy,**kw)
            result={'status':r['result']['status'],'output':r['out'],'benchmark':r['benchmark']}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result.get('benchmark',{}).get('api_error'): return 2
        return 0
    except PhaseError as e:
        print(json.dumps({'status':e.code,'details':e.details},ensure_ascii=False));return 2


if __name__=='__main__':
    raise SystemExit(main())
