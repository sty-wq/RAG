"""Read models on workflow tables. Disk inventory is sampled by dispatcher, never by dashboard reads."""
import collections,datetime,json,math,os,shutil,sqlite3,tempfile,time
from pathlib import Path
from .. import __version__
from ..common import PhaseError
from .store import Store,root

LABELS={'ADMISSION_REJECTED':'输入被拒绝','CREATED':'等待处理','QUEUED':'等待处理','PROCESSING':'正在识别','WAITING_REVIEW':'等待人工确认','COMPLETED':'已完成','PARTIAL':'部分完成','FAILED':'失败','UNSUPPORTED':'不支持的文档','PROVIDER_UNAVAILABLE':'OCR 服务不可用'}

def percent(n,d):return {'numerator':n,'denominator':d,'rate':n/d if d else None}
def quantiles(values):
    v=sorted(x for x in values if x is not None)
    return {'n':len(v),'p50':v[max(0,math.ceil(.5*len(v))-1)] if v else None,'p95':v[max(0,math.ceil(.95*len(v))-1)] if v else None,'min':min(v) if v else None,'max':max(v) if v else None}

def job_list(s,status=None,filename=None,since=None,until=None):
    where=[];args=[]
    for expr,value in [('j.status=?',status),('j.name LIKE ?',('%'+filename+'%') if filename else None),('j.created>=?',since),('j.created<=?',until)]:
        if value is not None:where.append(expr);args.append(value)
    return s.all('''SELECT j.*, (SELECT count(*) FROM review_items r WHERE r.job_id=j.id) AS review_count,
      (SELECT max(updated) FROM page_stages p WHERE p.job_id=j.id) AS page_updated,
      (SELECT max(updated) FROM page_stages p WHERE p.job_id=j.id)-j.created AS processing_elapsed_seconds
      FROM jobs j '''+('WHERE '+' AND '.join(where) if where else '')+' ORDER BY j.created DESC LIMIT 200',args)

def cloud(s,since=None,until=None,profile=None,strategy=None,status=None,error=None):
    where=[];args=[]
    for expr,val in [('started_at>=?',since),('started_at<=?',until),('profile=?',profile),('input_strategy=?',strategy),('status=?',status),('error_type=?',error)]:
        if val is not None:where.append(expr);args.append(val)
    clause=' WHERE '+' AND '.join(where) if where else ''
    rows=s.all('SELECT * FROM provider_attempts'+clause+' ORDER BY started_at DESC LIMIT 1000',args)
    # Aggregates cover ALL filtered records, not the display limit.
    counts=s.one('''SELECT count(*) AS invocations,
    sum(kind='OCR' AND cache_hit=0) AS ocr_calls,sum(kind='OAUTH') AS oauth_calls,
    sum(kind='OCR' AND cache_hit=0 AND status='SUCCESS') AS successful,
    sum(kind='OCR' AND cache_hit=0 AND status='FAILED') AS failed,
    sum(kind='OCR' AND cache_hit=0 AND status='BILLING_STATUS_UNKNOWN') AS unknown,
    sum(kind='OCR' AND cache_hit=0 AND status IN ('STARTED','HTTP_RECEIVED')) AS in_flight,
    sum(cache_hit) AS cache_hits,sum(mode='REPLAY' AND kind='OCR') AS replay_runs,
    sum(kind='OCR' AND retry_of IS NOT NULL) AS retried,sum(kind='OCR' AND mode='FORCE_LIVE') AS force_live,
    sum(kind='OCR') AS ocr_invocations,sum(estimated_billing_units) AS estimated_units
    FROM provider_attempts'''+clause,args)
    counts={k:int(v or 0) for k,v in counts.items()}
    latency=s.all('SELECT latency_ms FROM provider_attempts'+(clause+' AND ' if clause else ' WHERE ')+"kind='OCR' AND cache_hit=0 AND status='SUCCESS'",args)
    return {'counts':counts,'latency':quantiles([r['latency_ms'] for r in latency]),'success_rate':percent(counts['successful'],counts['ocr_calls']),
      'error_rate':percent(counts['failed'],counts['ocr_calls']),'cache_hit_rate':percent(counts['cache_hits'],counts['ocr_invocations']),
      'budget':s.stats(),'attempts':rows,'display_limit':1000,'provider_quota':'Unknown'}

def overview(s):
    now=datetime.datetime.now(datetime.timezone.utc);day=now.replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
    states={k:0 for k in LABELS};states.update({r['status']:r['n'] for r in s.all('SELECT status,count(*) n FROM jobs GROUP BY status')})
    stats=cloud(s,since=day)
    days=[(now-datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in reversed(range(7))]
    def series(sql):return {r['day']:r['n'] for r in s.all(sql,(day-6*86400,))}
    jobs=series("SELECT date(created,'unixepoch') day,count(*) n FROM jobs WHERE created>=? GROUP BY day")
    calls=series("SELECT date(started_at,'unixepoch') day,count(*) n FROM provider_attempts WHERE kind='OCR' AND cache_hit=0 AND started_at>=? GROUP BY day")
    reviews=series("SELECT date(timestamp,'unixepoch') day,count(*) n FROM decisions WHERE timestamp>=? GROUP BY day")
    latency=series("SELECT date(started_at,'unixepoch') day,avg(latency_ms) n FROM provider_attempts WHERE kind='OCR' AND cache_hit=0 AND status='SUCCESS' AND started_at>=? GROUP BY day")
    from ..admission import rejection_summary
    admission=rejection_summary(s.root)
    return {'admission':admission,'categories':{'Admission Rejected':admission['count'],'Provider Failed':states.get('PROVIDER_UNAVAILABLE',0),'Processing Failed':states.get('FAILED',0),'Completed':states.get('COMPLETED',0),'Partial':states.get('PARTIAL',0)},'states':states,'documents_today':s.one('SELECT count(*) n FROM jobs WHERE created>=?',(day,))['n'],
      'pages_today':s.one("SELECT count(*) n FROM stage_events WHERE scope='PAGE_CREATED' AND occurred>=?",(day,))['n'],
      'pending_reviews':s.one("SELECT count(*) n FROM review_items WHERE status='OPEN' AND review_requirement='REQUIRED'")['n'],
      'unresolved_reviews':s.one("SELECT count(*) n FROM review_items WHERE status='UNRESOLVED'")['n'],
      'completion_rate':percent(states.get('COMPLETED',0),sum(v for k,v in states.items() if k!='ADMISSION_REJECTED')),
      'cloud_today':stats,'trends':[{'day':d,'jobs':jobs.get(d,0),'calls':calls.get(d,0),'reviews':reviews.get(d,0),'mean_latency':latency.get(d)} for d in days],'timezone':'UTC'}

def review_analytics(s):
    states={r['status']:r['n'] for r in s.all('SELECT status,count(*) n FROM review_items GROUP BY status')}
    forms={r['selected_form']:r['n'] for r in s.all('SELECT selected_form,count(*) n FROM decisions GROUP BY selected_form')}
    docs=s.one('SELECT count(*) n FROM jobs')['n'];reviewdocs=s.one('SELECT count(DISTINCT job_id) n FROM review_items')['n']
    return {'target_counts':s.all('SELECT target_kind,review_requirement,status,count(*) n FROM review_items GROUP BY target_kind,review_requirement,status'),'states':states,'forms':forms,'review_rate':percent(reviewdocs,docs),'unresolved_rate':percent(states.get('UNRESOLVED',0),sum(states.values())),
      'confusions':s.all('SELECT observed,confirmed,count(*) n FROM confusion_events GROUP BY observed,confirmed ORDER BY n DESC LIMIT 100'),'active_review_time':None}

def timeline(s,j):
    job=s.job(j)
    events=s.all('SELECT * FROM stage_events WHERE job_id=? ORDER BY occurred,id',(j,))
    return {'job':job,'events':events,'pages':s.all('SELECT * FROM page_stages WHERE job_id=? ORDER BY page_index',(j,)),
      'attempts':s.all('SELECT * FROM provider_attempts WHERE document_id=? ORDER BY started_at',(j,)),
      'reviews':s.all('SELECT id,page_index,cell_id,target_kind,target_id,review_requirement,status,revision FROM review_items WHERE job_id=? ORDER BY page_index,cell_id',(j,)),
      'decisions':s.all('SELECT d.* FROM decisions d JOIN review_items r ON d.review_id=r.id WHERE r.job_id=? ORDER BY timestamp',(j,)),
      'history_note':'阶段历史从 schema v2 起真实记录；旧任务只有原已持久化时间，不补造中间阶段。'}

def errors(s):
    rows=s.all('''SELECT error code,count(*) n,min(updated) first_seen,max(updated) last_seen,job_id,page_index
      FROM page_stages WHERE error IS NOT NULL GROUP BY error,job_id,page_index ORDER BY last_seen DESC LIMIT 200''')
    for r in s.all('''SELECT coalesce(error_type,status) code,count(*) n,min(started_at) first_seen,max(started_at) last_seen,document_id job_id,page_id page_index
      FROM provider_attempts WHERE cache_hit=0 AND status NOT IN ('SUCCESS','CACHE_HIT') GROUP BY code,document_id,page_id ORDER BY last_seen DESC LIMIT 200'''):rows.append(r)
    return rows

def benchmark(s):
    rows=s.all('SELECT * FROM benchmark_runs ORDER BY recorded DESC LIMIT 20')
    for r in rows:r['metrics']=json.loads(r['metrics'])
    return rows

def structure(s):
    stages=s.all('SELECT status,error,count(*) n FROM page_stages GROUP BY status,error')
    corpus=[]
    p=Path(os.environ.get('CABLE_BENCHMARK_DIR','artifacts/phase02_benchmark'))/'results.json'
    try:
        corpus=[{'id':r['id'],'group':r['group'],'status':r.get('result_status'),'reason':r.get('api_error'),'gt_source':r.get('gt_source')} for r in json.loads(p.read_text())]
    except (OSError,ValueError):pass
    counts=s.all("SELECT o.reason,count(*) n FROM structure_observations o JOIN jobs j ON j.id=o.job_id WHERE j.processing_contract='V2' GROUP BY o.reason")
    total=sum(x['n'] for x in counts);passed=sum(x['n'] for x in counts if x['reason']=='PRE_PASSED')
    return {'page_states':stages,'reasons':counts,'success_rate':percent(passed,total),'review_rate':percent(total-passed,total),'corpus':corpus}

def health(data_root=None):
    path=Path(data_root or root());checks={}
    try:
        with sqlite3.connect(f'file:{path}/workflow.sqlite3?mode=rw',uri=True,timeout=.5) as c:c.execute('SELECT 1 FROM jobs LIMIT 1').fetchone()
        checks['sqlite']=True
    except Exception:checks['sqlite']=False
    try:
        with tempfile.TemporaryFile(dir=path) as f:f.write(b'health');f.flush();os.fsync(f.fileno())
        checks['artifacts']=True
    except Exception:checks['artifacts']=False
    try:
        import redis
        r=redis.Redis.from_url(os.environ.get('CABLE_BROKER_URL','redis://127.0.0.1:6380/0'),socket_connect_timeout=.5,socket_timeout=.5)
        checks['valkey']=bool(r.ping());r.close()
    except Exception:checks['valkey']=False
    return {'ready':all(checks.values()),'checks':checks,'cloud_hard_dependency':False}

def system(s):
    h=health(s.root);disk=shutil.disk_usage(s.root)
    snapshot=s.one("SELECT * FROM operational_snapshots WHERE name='artifact_inventory'")
    return {'health':h,'disk_free_bytes':disk.free,'disk_total_bytes':disk.total,'app_version':__version__,'schema_version':s.one('PRAGMA user_version')['user_version'],
      'heartbeats':s.all('SELECT role,updated FROM runtime_heartbeats'),'now':time.time(),
      'artifact_inventory':json.loads(snapshot['value']) if snapshot else None,'inventory_updated':snapshot['recorded'] if snapshot else None,
      'retention_enabled':False,'local_ocr':False,'known_provider_quota':'Unknown'}

def collect_artifacts(s,deadline=2):
    start=time.monotonic();count=total=0;oldest=None;largest=collections.Counter();partial=False
    roots=[s.root/part for part in ['jobs','uploads']]
    roots.append(Path(os.environ.get('CABLE_CACHE_DIR','artifacts/request_cache')))
    for folder in roots:
        for base,dirs,files in os.walk(folder,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
            if time.monotonic()-start>deadline:partial=True;break
            for name in files:
                p=Path(base)/name
                try:
                    st=p.stat();count+=1;total+=st.st_size;oldest=min(oldest or st.st_mtime,st.st_mtime)
                    if p.is_relative_to(s.root/'jobs'):largest[p.relative_to(s.root/'jobs').parts[0]]+=st.st_size
                except OSError:pass
        if partial:break
    value={'file_count':count,'bytes':total,'oldest_mtime':oldest,'largest_jobs':largest.most_common(10),'partial_scan':partial,'elapsed_ms':(time.monotonic()-start)*1000}
    with s.tx() as c:c.execute('INSERT OR REPLACE INTO operational_snapshots VALUES(?,?,?)',('artifact_inventory',json.dumps(value),time.time()))
