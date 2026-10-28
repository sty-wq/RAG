"""Cross-process cache lock + durable budget reservation. No automatic paid retry.

Ambiguous sends require an explicit FORCE_LIVE reason from the experiment CLI.
An already durable response is always replayed during normal recovery.
"""
import contextlib
import fcntl
import json
import time
import signal
import threading
from pathlib import Path
import requests
from ..baidu import BaiduCloudOCRProvider, request_spec, ENDPOINT, TOKEN_ENDPOINT, PROFILE
from ..common import PhaseError, save_json
from .store import ident


@contextlib.contextmanager
def file_lock(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:yield False;return
        try:yield True
        finally:fcntl.flock(f,fcntl.LOCK_UN)


class AuditedSession:
    def __init__(self,store,context,session=None):
        self.store=store;self.context=context;self.session=session
        self.attempt_ids=[]

    def post(self,url,**kwargs):
        s=self.store;ctx=self.context;aid=ident();now=time.time()
        if self.session is None:self.session=requests.Session()
        kind='OCR' if url==ENDPOINT else 'OAUTH'
        # The only caller is the fixed Baidu transport; never save params, payload, headers or response tokens.
        with s.tx() as c:
            if kind=='OCR':
                # -1 explicitly disables the internal ceiling; reservations remain durable.
                updated=c.execute('UPDATE budget SET reserved=reserved+1 WHERE id=1 AND (max_calls=-1 OR reserved<max_calls)').rowcount
                if not updated:raise PhaseError('API_BUDGET_EXHAUSTED')
            c.execute('''INSERT INTO provider_attempts(id,provider,endpoint,profile,document_id,page_id,input_strategy,
              input_hash,request_key,started_at,status,estimated_billing_units,retry_of,kind,mode,billing_status)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              (aid,'baidu',url,PROFILE,ctx.get('document_id'),str(ctx.get('page_id')),ctx['spec']['input_strategy'],
               ctx['spec']['input_hash'],ctx['key'],now,'STARTED',int(kind=='OCR'),ctx.get('retry_of'),kind,ctx['mode'],
               'BILLING_STATUS_UNKNOWN' if kind=='OCR' else 'NOT_OCR'))
        if ctx.get('representation'):
            save_json(s.root/'provider_attempt_metadata'/f'{aid}.json',{'attempt_id':aid,'request_key':ctx['key'],'representation':ctx['representation']})
        self.attempt_ids.append(aid)
        try:
            # Unix worker/CLI hard wall deadline, in addition to connect/read timeouts.
            # OCR is never invoked inside an API request thread.
            if threading.current_thread() is threading.main_thread():
                previous_handler=signal.getsignal(signal.SIGALRM)
                def timeout_handler(signum,frame):raise TimeoutError('provider deadline')
                signal.signal(signal.SIGALRM,timeout_handler)
                previous_timer=signal.setitimer(signal.ITIMER_REAL,110)
                try:response=self.session.post(url,**kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,*previous_timer)
                    signal.signal(signal.SIGALRM,previous_handler)
            else:
                raise PhaseError('PROVIDER_REQUIRES_WORKER_MAIN_THREAD')
            # Raw persistence is handled immediately by base provider after return.
            try:raw=response.json()
            except Exception:raw={}
            code=raw.get('error_code') if isinstance(raw,dict) else None
            status=('SUCCESS' if kind=='OAUTH' and isinstance(raw,dict) and isinstance(raw.get('access_token'),str) else 'HTTP_RECEIVED') if response.status_code==200 and not code else 'FAILED'
            err='QUOTA_ERROR' if code in {17,18,19} else 'INVALID_CREDENTIAL' if code in {110,111,100} else f'PROVIDER_{code}' if code else None
            with s.tx() as c:
                c.execute('UPDATE provider_attempts SET finished_at=?,latency_ms=?,status=?,provider_request_id=?,error_type=? WHERE id=?',
                   (time.time(),(time.time()-now)*1000,status,str(raw.get('log_id')) if isinstance(raw,dict) and raw.get('log_id') else None,err,aid))
            return response
        except Exception as e:
            with s.tx() as c:
                c.execute('UPDATE provider_attempts SET finished_at=?,latency_ms=?,status=?,error_type=? WHERE id=?',
                          (time.time(),(time.time()-now)*1000,'BILLING_STATUS_UNKNOWN' if kind=='OCR' else 'FAILED',type(e).__name__,aid))
            raise


class AccountedProvider:
    def __init__(self,store,env_path,cache_dir,document_id=None,page_id=None,session=None,force_reason=None):
        self.store=store;self.env_path=env_path;self.cache_dir=Path(cache_dir)
        self.document_id=document_id;self.page_id=page_id;self.session=session;self.force_reason=force_reason

    def set_representation(self,metadata):
        self.representation=metadata

    def recognize_page(self,input_bytes,strategy,page_index=0,mode='REPLAY'):
        if mode not in {'LIVE','REPLAY','FORCE_LIVE'}:raise PhaseError('INVALID_EXECUTION_MODE')
        spec,key=request_spec(input_bytes,strategy,page_index)
        if mode=='FORCE_LIVE' and not self.force_reason:raise PhaseError('FORCE_LIVE_REASON_REQUIRED')
        s=self.store
        with file_lock(self.cache_dir/'locks'/f'{key}.lock'):
            cached=(self.cache_dir/key/'request.json').exists()
            previous=s.one("SELECT * FROM provider_attempts WHERE request_key=? AND kind='OCR' AND cache_hit=0 ORDER BY started_at DESC LIMIT 1",(key,))
            if not cached and previous and mode!='FORCE_LIVE':
                # Request could have reached provider; never silently re-charge on redelivery.
                raise PhaseError('BILLING_STATUS_UNKNOWN' if previous['status'] in {'STARTED','HTTP_RECEIVED','BILLING_STATUS_UNKNOWN'} else 'PREVIOUS_PROVIDER_FAILURE')
            if mode!='REPLAY' and (not cached or mode=='FORCE_LIVE'):
                budget=s.one('SELECT max_calls,reserved FROM budget WHERE id=1')
                if budget['max_calls']!=-1 and budget['reserved']>=budget['max_calls']:raise PhaseError('API_BUDGET_EXHAUSTED')
            context={'document_id':self.document_id,'page_id':self.page_id,'spec':spec,'key':key,'mode':mode,
                     'representation':getattr(self,'representation',None),'retry_of':previous['id'] if previous and mode=='FORCE_LIVE' else None}
            cache=self.cache_dir
            if mode=='FORCE_LIVE':
                cache=self.cache_dir/'force_runs'/ident()
                save_json(cache/'experiment.json',{'reason':self.force_reason,'request_key':key,'created':time.time()})
            session=AuditedSession(s,context,self.session)
            provider=BaiduCloudOCRProvider(self.env_path,cache,session=session)
            start=time.time()
            try:
                result=provider.recognize_page(input_bytes,strategy,page_index,'LIVE' if mode=='FORCE_LIVE' else mode)
            except PhaseError as e:
                with s.tx() as c:
                    for aid in session.attempt_ids:
                        c.execute("UPDATE provider_attempts SET status=CASE WHEN status IN ('STARTED','HTTP_RECEIVED') THEN 'FAILED' ELSE status END,error_type=COALESCE(error_type,?) WHERE id=?",(e.code,aid))
                raise
            with s.tx() as c:
                for aid in session.attempt_ids:
                    c.execute("UPDATE provider_attempts SET status='SUCCESS',billing_status=CASE WHEN kind='OCR' THEN 'ESTIMATED_NOT_RECONCILED' ELSE 'NOT_OCR' END WHERE id=?",(aid,))
                if result['cache_hit']:
                    c.execute('''INSERT INTO provider_attempts(id,provider,endpoint,profile,document_id,page_id,input_strategy,input_hash,
                      request_key,started_at,finished_at,latency_ms,status,estimated_billing_units,cache_hit,kind,mode,billing_status)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0,1,'OCR',?,'CACHE_NO_CHARGE')''',
                      (ident(),'baidu',ENDPOINT,PROFILE,self.document_id,str(self.page_id),strategy,spec['input_hash'],key,start,time.time(),
                       (time.time()-start)*1000,'CACHE_HIT',mode))
                    c.execute("UPDATE provider_attempts SET status='SUCCESS',billing_status='RECOVERED_DURABLE_RESPONSE' WHERE request_key=? AND kind='OCR' AND status IN ('STARTED','HTTP_RECEIVED','BILLING_STATUS_UNKNOWN')",(key,))
            if mode=='FORCE_LIVE' and not cached:
                import shutil
                # Under the same request lock; first known success becomes default replay evidence.
                target=self.cache_dir/key
                target.mkdir(parents=True,exist_ok=True)
                from ..common import save_bytes
                for name in ('response.bin','response.json','request.json'):
                    save_bytes(target/name,(cache/key/name).read_bytes())
            result['execution_mode']=mode
            return result
