import json,os
from pathlib import Path
from ..common import PhaseError

def switches():
    path=Path(os.environ.get('CABLE_CONFIG_FILE','config.json'))
    try:config=json.loads(path.read_text()) if path.is_file() else {'ocr':{'local':{'enabled':False},'cloud':{'enabled':True,'provider':'baidu'}}}
    except (OSError,ValueError):raise PhaseError('INVALID_CONFIG') from None
    ocr=config.get('ocr',{})
    if ocr.get('local',{}).get('enabled',False):raise PhaseError('LOCAL_OCR_NOT_IMPLEMENTED')
    provider=ocr.get('cloud',{}).get('provider','baidu')
    if provider not in {'baidu','deepseek'}:raise PhaseError('UNSUPPORTED_CLOUD_PROVIDER')
    return {'local':False,'cloud':bool(ocr.get('cloud',{}).get('enabled',True)),'provider':provider}
