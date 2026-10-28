"""Accounted DeepSeek image transcription with isolated cache and no paid retry."""
import base64
import io
import json
import os
import signal
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values
from PIL import Image

from ..common import PhaseError, digest, save_bytes, save_json, stable_hash
from ..deepseek_schema import strict_parse
from .provider import file_lock
from .store import ident

PROFILE='vision_transcription_p5_v1'
PARAMETERS={'temperature':0,'max_tokens':32768,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},'stream':False}
PROMPT=Path(__file__).parents[1]/'prompts/deepseek_ocr_p5.txt'

def configuration(env_path,require_key=True):
    values={**dotenv_values(env_path),**os.environ}
    base=values.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')
    model=values.get('DEEPSEEK_MODEL','deepseek-v4-flash-vision-exp')
    key=values.get('DEEPSEEK_API_KEY','')
    url=urlsplit(base)
    if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise PhaseError('DEEPSEEK_CONFIGURATION_INVALID')
    if not model or (require_key and not key):raise PhaseError('DEEPSEEK_MISSING_CREDENTIALS')
    try:timeout=int(values.get('DEEPSEEK_TIMEOUT_SECONDS','180'))
    except (ValueError,TypeError):raise PhaseError('DEEPSEEK_CONFIGURATION_INVALID') from None
    if not 5<=timeout<=210:raise PhaseError('DEEPSEEK_CONFIGURATION_INVALID')
    return {'endpoint':base+'/chat/completions','model':model,'key':key,'timeout':timeout}

def request_identity(data,strategy,page_index,cfg):
    spec={'provider':'deepseek','endpoint':cfg['endpoint'],'model':cfg['model'],'profile':PROFILE,
          'input_hash':digest(data),'input_strategy':strategy,'page_index':page_index,
          'prompt_sha256':digest(PROMPT.read_bytes()),'parameters':PARAMETERS,'image_detail':'original'}
    return spec,stable_hash(spec)

def decode_response(raw,http_status):
    if http_status!=200:raise PhaseError('DEEPSEEK_HTTP_'+str(http_status))
    try:
        envelope=json.loads(raw);choice=envelope['choices'][0];content=choice['message']['content']
        if choice.get('finish_reason')=='length':raise PhaseError('DEEPSEEK_OUTPUT_TRUNCATED')
        if choice.get('finish_reason')!='stop':raise PhaseError('DEEPSEEK_INCOMPLETE_RESPONSE')
        if not isinstance(content,str):raise PhaseError('DEEPSEEK_INVALID_RESPONSE')
        value,valid=strict_parse(content)
        if not valid['schema_valid']:raise PhaseError('DEEPSEEK_'+valid['errors'][0])
        atoms=value['lines'] if value['type']=='lines' else [c for r in value['rows'] for c in r['cells']]+value.get('outside_table_lines',[])
        if len(atoms)>12000 or sum(len(a['text']) for a in atoms)>500000:raise PhaseError('DEEPSEEK_OUTPUT_LIMIT')
        if value['type']=='table' and (not value['rows'] or any(not r['cells'] for r in value['rows'])):raise PhaseError('DEEPSEEK_SCHEMA_MISMATCH')
        return value,envelope
    except PhaseError:raise
    except (ValueError,KeyError,IndexError,TypeError):raise PhaseError('DEEPSEEK_INVALID_RESPONSE') from None

class DeepSeekProvider:
    def __init__(self,store,env_path,cache_dir,document_id=None,page_id=None,session=None):
        self.store=store;self.env_path=env_path;self.cache_dir=Path(cache_dir)/'deepseek'
        self.document_id=document_id;self.page_id=page_id;self.session=session

    def recognize_page(self,data,strategy,page_index=0,mode='REPLAY'):
        if mode not in {'LIVE','REPLAY'}:raise PhaseError('INVALID_EXECUTION_MODE')
        cfg=configuration(self.env_path,False);spec,key=request_identity(data,strategy,page_index,cfg)
        with Image.open(io.BytesIO(data)) as image:
            if max(image.size)>8192:raise PhaseError('DEEPSEEK_INPUT_DIMENSION_EXCEEDED')
            mime={'JPEG':'image/jpeg','PNG':'image/png'}.get(image.format)
        if not mime or len(data)>32*1024*1024:raise PhaseError('DEEPSEEK_INPUT_SIZE_INVALID')
        with file_lock(self.cache_dir/'locks'/(key+'.lock')):
            folder=self.cache_dir/key;record=folder/'request.json';s=self.store;start=time.time();cached=record.exists();aid=ident()
            if cached:
                try:
                    meta=json.loads(record.read_text());raw=(folder/'response.bin').read_bytes()
                    if meta['spec']!=spec or digest(raw)!=meta['raw_sha256']:raise ValueError()
                    status=meta['http_status']
                except (OSError,ValueError,KeyError):raise PhaseError('DEEPSEEK_CACHE_INVALID') from None
            else:
                if mode=='REPLAY':raise PhaseError('DEEPSEEK_CACHE_MISS')
                previous=s.one("SELECT id FROM provider_attempts WHERE provider='deepseek' AND request_key=? AND cache_hit=0 LIMIT 1",(key,))
                if previous:raise PhaseError('PREVIOUS_REQUEST_OUTCOME_UNKNOWN')
                cfg=configuration(self.env_path,True)
                if threading.current_thread() is not threading.main_thread():raise PhaseError('PROVIDER_REQUIRES_WORKER_MAIN_THREAD')
                body={'model':cfg['model'],**PARAMETERS,'messages':[{'role':'user','content':[
                    {'type':'text','text':PROMPT.read_text()},
                    {'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(data).decode(),'detail':'original'}}]}]}
                encoded=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
                if len(encoded)>48*1024*1024:raise PhaseError('DEEPSEEK_INPUT_SIZE_INVALID')
            with s.tx() as c:
                if not cached:
                    if not c.execute('UPDATE budget SET reserved=reserved+1 WHERE id=1 AND (max_calls=-1 OR reserved<max_calls)').rowcount:raise PhaseError('API_BUDGET_EXHAUSTED')
                c.execute('''INSERT INTO provider_attempts(id,provider,endpoint,profile,document_id,page_id,input_strategy,input_hash,
                    request_key,started_at,status,estimated_billing_units,cache_hit,kind,mode,billing_status)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'OCR',?,?)''',
                    (aid,'deepseek',cfg['endpoint'],PROFILE,self.document_id,str(self.page_id),strategy,spec['input_hash'],key,start,
                     'STARTED',0 if cached else 1,int(cached),mode,'CACHE_NO_CHARGE' if cached else 'BILLING_STATUS_UNKNOWN'))
            try:
                if not cached:
                    session=self.session or requests.Session()
                    if self.session is None:session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
                    previous_handler=signal.getsignal(signal.SIGALRM)
                    def deadline(*_):raise TimeoutError()
                    signal.signal(signal.SIGALRM,deadline);previous_timer=signal.setitimer(signal.ITIMER_REAL,cfg['timeout'])
                    try:
                        response=session.post(cfg['endpoint'],data=encoded,headers={'Authorization':'Bearer '+cfg['key'],'Content-Type':'application/json'},timeout=(15,min(90,cfg['timeout'])),allow_redirects=False)
                        raw=response.content;status=response.status_code
                    finally:
                        signal.setitimer(signal.ITIMER_REAL,*previous_timer);signal.signal(signal.SIGALRM,previous_handler)
                        if self.session is None:session.close()
                    # Never persist a credential echoed by an upstream failure.
                    raw=raw.replace(cfg['key'].encode(),b'[REDACTED_API_KEY]')
                    save_bytes(folder/'response.bin',raw)
                    save_json(record,{'spec':spec,'raw_sha256':digest(raw),'http_status':status,'request_body_sha256':digest(encoded),'automatic_retries':0})
                value,envelope=decode_response(raw,status)
                usage=envelope.get('usage') or {}
                usage={k:v for k,v in usage.items() if k in {'prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'} and type(v) is int}
                with s.tx() as c:
                    c.execute('UPDATE provider_attempts SET status=?,finished_at=?,latency_ms=?,provider_request_id=?,reported_billing_info=?,billing_status=? WHERE id=?',
                        ('CACHE_HIT' if cached else 'SUCCESS',time.time(),(time.time()-start)*1000,str(envelope.get('id',''))[:200],json.dumps(usage),'CACHE_NO_CHARGE' if cached else 'USAGE_REPORTED_COST_UNVERIFIED',aid))
                return {'provider':'deepseek','profile':PROFILE,'request_key':key,'document':value,'raw_response_ref':str(folder/'response.bin'),
                        'cache_hit':cached,'model':cfg['model'],'execution_mode':mode,'new_ocr_http_calls':int(not cached),'usage':usage}
            except Exception as exc:
                code=exc.code if isinstance(exc,PhaseError) else 'DEEPSEEK_TIMEOUT' if isinstance(exc,(TimeoutError,requests.Timeout)) else 'DEEPSEEK_NETWORK_ERROR' if isinstance(exc,requests.RequestException) else 'DEEPSEEK_PROCESSING_ERROR'
                unknown=not cached and not record.exists()
                with s.tx() as c:c.execute('UPDATE provider_attempts SET status=?,error_type=?,finished_at=?,latency_ms=? WHERE id=?',('BILLING_STATUS_UNKNOWN' if unknown else 'FAILED',code,time.time(),(time.time()-start)*1000,aid))
                raise PhaseError(code) from None
