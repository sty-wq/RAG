import datetime,json,os,time,re
from pathlib import Path
from typing import Literal
from fastapi import APIRouter,Request,File,UploadFile,Form
from fastapi.responses import HTMLResponse,JSONResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from ..common import PhaseError
from ..admission import MESSAGES,read_upload
from ..ir import FIELDS
from .store import Store
from .settings import switches
from . import observability as obs,access
router=APIRouter();templates=Jinja2Templates(directory=Path(__file__).parent/'templates')
CHANNEL_LABELS={'baidu':'百度 OCR','deepseek':'DeepSeek'}
CHANNEL_MESSAGES={
    'DEEPSEEK_OUTPUT_TRUNCATED':'DeepSeek 输出达到上限，未生成完整结果。可切换百度 OCR 重新上传。',
    'DEEPSEEK_INVALID_JSON':'DeepSeek 返回的内容无法解析。可切换百度 OCR 重新上传。',
    'DEEPSEEK_SCHEMA_MISMATCH':'DeepSeek 返回的表格或行结构不完整。可切换百度 OCR 重新上传。',
    'DEEPSEEK_MISSING_CREDENTIALS':'DeepSeek 尚未配置，请联系管理员或选择百度 OCR。',
    'DEEPSEEK_CONFIGURATION_INVALID':'DeepSeek 配置不可用，请联系管理员。',
    'DEEPSEEK_TIMEOUT':'DeepSeek 响应超时。可切换百度 OCR 重新上传。',
    'DEEPSEEK_NETWORK_ERROR':'DeepSeek 连接失败。可稍后检查结果或选择百度 OCR。',
    'DEEPSEEK_INPUT_DIMENSION_EXCEEDED':'图片超过 DeepSeek 的尺寸限制，请选择其他通道或较小的图片。',
}
templates.env.filters['ts']=lambda x:datetime.datetime.fromtimestamp(x,datetime.timezone.utc).strftime('%m-%d %H:%M:%S') if x else '—'
templates.env.filters['pct']=lambda x:f"{x['rate']:.1%} ({x['numerator']}/{x['denominator']})" if x and x.get('rate') is not None else 'N/A'
templates.env.filters['metric']=lambda x:f"{x['rate']:.1%} ({x.get('correct',0)}/{x.get('total',0)})" if x and x.get('rate') is not None else 'N/A'

def page(request,name,**data):
    return templates.TemplateResponse(request=request,name=name,context=data)

def date_value(value):
    if not value:return None
    try:return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:raise PhaseError('INVALID_TIME_FILTER') from None

def public_job(s,j):
    job=s.job(j);pages=s.all('SELECT page_index,status FROM page_stages WHERE job_id=? ORDER BY page_index',(j,))
    processed=sum(p['status'] not in {'QUEUED','PROCESSING'} for p in pages)
    active=next((p['page_index']+1 for p in pages if p['status']=='PROCESSING'),None)
    result=s.read(job['result_ref']) if job['result_ref'] else {}
    if job['processing_contract']=='V3':
        from .public_v3 import public_result
        return {'id':j,'filename':job['name'],'status':job['status'],'status_label':obs.LABELS.get(job['status'],'处理中'),'created':job['created'],'page_count':job['page_count'],'processed_pages':processed,'current_page':active,'waiting_review':job['status']=='WAITING_REVIEW','ocr_provider':job['ocr_provider'],'ocr_provider_label':CHANNEL_LABELS[job['ocr_provider']],'channel_message':next((CHANNEL_MESSAGES[p['error']] for p in result.get('pages',[]) if p.get('error') in CHANNEL_MESSAGES),None),'admission_message':next((MESSAGES[p['error']] for p in result.get('pages',[]) if p.get('error') in MESSAGES),None),**public_result(result)}
    unresolved=[];seen=set()
    positions={}
    for index,links in enumerate(result.get('field_sources',[])):
        if isinstance(links,list):
            for field,link in zip(FIELDS,links):positions[(link.get('page_index',0),link.get('cell_id'))]=(index,field)
    for u in result.get('unresolved',[]):
        key=(u.get('page',0),u.get('cell_id'))
        if key in seen:continue
        seen.add(key)
        pos=positions.get((u.get('page',0),u.get('cell_id')))
        match=re.search(r'-r(\d+)-c(\d+)$',u.get('cell_id',''))
        row=int(match.group(1)) if match else None
        field=FIELDS[int(match.group(2))] if match and int(match.group(2))<len(FIELDS) else None
        unresolved.append({'page':u.get('page',0)+1,'row':row,'field':field,'row_index':pos[0] if pos else None,
                           'message':'此位置尚未确认，结果不完整'})
    # No raw provider payloads, candidate IDs, file paths, debugging reason codes or billing fields.
    return {'id':j,'filename':job['name'],'status':job['status'],'status_label':obs.LABELS.get(job['status'],'处理中'),
      'created':job['created'],'page_count':job['page_count'],'processed_pages':processed,'current_page':active,
      'waiting_review':job['status']=='WAITING_REVIEW','rows':result.get('rows',[]) if job['status'] in {'COMPLETED','PARTIAL'} else [],
      'unresolved':unresolved if job['status']=='PARTIAL' else [],'confirmed':job['status']=='COMPLETED'}

@router.get('/login',response_class=HTMLResponse)
def login(request:Request):return page(request,'login.html',error=None)
@router.post('/login')
async def login_submit(request:Request,username:str=Form(...),password:str=Form(...)):
    v=access.settings();role=access.check(username,password,v)
    if not role:return page(request,'login.html',error='账号或密码不正确，或访问配置不可用')
    response=RedirectResponse('/admin' if role=='admin' else '/app',303)
    response.set_cookie('cable_test_session',access.token(role,username,v),httponly=True,samesite='strict',secure=os.environ.get('CABLE_COOKIE_SECURE')=='1',max_age=3600)
    return response
@router.post('/logout')
def logout():
    r=RedirectResponse('/login',303);r.delete_cookie('cable_test_session');return r

@router.get('/health/live')
def live():return {'alive':True}
@router.get('/health/ready')
def ready():
    h=obs.health();return JSONResponse(h,200 if h['ready'] else 503)

@router.get('/app',response_class=HTMLResponse)
def user_home(request:Request):
    s=Store();h=obs.health(s.root)
    try:config=switches();enabled=config['cloud'];default_provider=config['provider']
    except PhaseError:enabled=False;default_provider='baidu'
    last=s.one("SELECT status FROM provider_attempts WHERE provider=? AND kind='OCR' AND cache_hit=0 ORDER BY started_at DESC LIMIT 1",(default_provider,))
    provider='已启用，最近调用成功' if enabled and last and last['status']=='SUCCESS' else '已启用，最近调用未成功' if enabled and last else '已启用，尚无近期验证' if enabled else '未启用'
    from .deepseek_provider import configuration
    from .workflow import env_path
    try:configuration(env_path());deepseek_ready=True
    except PhaseError:deepseek_ready=False
    return page(request,'user_home.html',available=h['ready'],provider=provider,default_provider=default_provider,deepseek_ready=deepseek_ready)

@router.post('/app/documents',status_code=202)
async def user_upload(file:UploadFile=File(...),ocr_provider:Literal['baidu','deepseek']|None=Form(None)):
    s=Store();data=await read_upload(file,s.root);j=s.upload(data,file.filename or '',mode='LIVE',processing_contract='V3',ocr_provider=ocr_provider)
    return {'id':j['id'],'url':'/app/jobs/'+j['id']}
@router.get('/app/jobs/{j}',response_class=HTMLResponse)
def user_job(request:Request,j:str):return page(request,'user_job.html',job=public_job(Store(),j),fields=FIELDS)
@router.get('/app/api/jobs/{j}')
def user_status(j:str):return public_job(Store(),j)
@router.get('/app/jobs/{j}/result.json')
def user_result(j:str,download:bool=False):
    job=public_job(Store(),j);result={k:v for k,v in job.items() if k in ['status','rows','unresolved','confirmed','schema_version','parse_mode','review_status','pages','ocr_provider','ocr_provider_label']}
    return JSONResponse(result,headers={'Content-Disposition':'attachment; filename="result.json"'} if download else None)

@router.get('/admin',response_class=HTMLResponse)
def admin_home(request:Request):return page(request,'admin_overview.html',data=obs.overview(Store()),section='overview')
@router.get('/admin/jobs',response_class=HTMLResponse)
def admin_jobs(request:Request,status:str|None=None,filename:str|None=None,since:str|None=None,until:str|None=None):
    return page(request,'admin_jobs.html',jobs=obs.job_list(Store(),status or None,filename or None,date_value(since),date_value(until)),section='jobs')
@router.get('/admin/jobs/{j}',response_class=HTMLResponse)
def admin_job(request:Request,j:str):
    data=obs.timeline(Store(),j)
    history=list(data['events'])
    history.extend({'occurred':d['timestamp'],'scope':'Decision','page_index':None,'stage':d['actor']+' / '+d['selected_form']} for d in data['decisions'])
    history.extend({'occurred':a['started_at'],'scope':'Provider start','page_index':None,'stage':a['kind']+' / '+a['mode']} for a in data['attempts'])
    data['history']=sorted(history,key=lambda x:x['occurred'])
    return page(request,'admin_job.html',data=data,section='jobs')
@router.get('/admin/cloud',response_class=HTMLResponse)
def admin_cloud(request:Request,since:str|None=None,until:str|None=None,profile:str|None=None,strategy:str|None=None,status:str|None=None,error:str|None=None):
    return page(request,'admin_cloud.html',data=obs.cloud(Store(),date_value(since),date_value(until),profile or None,strategy or None,status or None,error or None),section='cloud')
@router.get('/admin/review-analytics',response_class=HTMLResponse)
def admin_reviews(request:Request):return page(request,'admin_reviews.html',data=obs.review_analytics(Store()),section='review')
@router.get('/admin/structure',response_class=HTMLResponse)
def admin_structure(request:Request):
    from .metrics_v3 import metrics
    s=Store();return page(request,'admin_structure.html',data=obs.structure(s),v3=metrics(s),section='structure')
@router.get('/admin/errors',response_class=HTMLResponse)
def admin_errors(request:Request):return page(request,'admin_errors.html',errors=obs.errors(Store()),section='errors')
@router.get('/admin/benchmark',response_class=HTMLResponse)
def admin_benchmark(request:Request):return page(request,'admin_benchmark.html',runs=obs.benchmark(Store()),section='benchmark')
@router.get('/admin/system',response_class=HTMLResponse)
def admin_system(request:Request):return page(request,'admin_system.html',data=obs.system(Store()),section='system')
