import json
import os
import sqlite3
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,File,UploadFile,Request,Form,Header
from fastapi.responses import JSONResponse,HTMLResponse,FileResponse,RedirectResponse,Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from ..common import PhaseError
from ..admission import MESSAGES,ADMISSION_CODES,read_upload,log_rejection
from .store import Store
from .workflow import review,decide

app=FastAPI(title='Cable OCR · V3')
app.add_middleware(TrustedHostMiddleware,allowed_hosts=os.environ.get('CABLE_ALLOWED_HOSTS','127.0.0.1,localhost,testserver').split(','))
templates=Jinja2Templates(directory=Path(__file__).parent/'templates')

@app.middleware('http')
async def local_boundary(request,call_next):
    origin=request.headers.get('origin')
    if request.method=='POST' and origin and origin!=str(request.base_url).rstrip('/'):
        return JSONResponse({'error':'ORIGIN_REJECTED'},403)
    length=request.headers.get('content-length','0')
    try:too_big=int(length)>26*1024*1024
    except ValueError:too_big=True
    if too_big:
        log_rejection(Store().root,'FILE_TOO_LARGE',{'request_bytes':length[:20]})
        return JSONResponse({'error':'FILE_TOO_LARGE','message':MESSAGES['FILE_TOO_LARGE']},413)
    from . import access
    path=request.url.path
    if access.required() and path not in {'/health/live','/login','/favicon.ico'}:
        cfg=access.settings()
        if not cfg:return JSONResponse({'error':'ACCESS_CONFIGURATION_MISSING'},503)
        who=access.principal(request,cfg)
        if not who:
            if 'text/html' in request.headers.get('accept',''):return RedirectResponse('/login',303)
            return JSONResponse({'error':'AUTHENTICATION_REQUIRED'},401,headers={'WWW-Authenticate':'Basic realm="Cable OCR Test"'})
        if who['role']!='admin' and not access.user_area(path) and path!='/logout':return JSONResponse({'error':'ADMIN_REQUIRED'},403)
        request.state.principal=who
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Cache-Control']='no-store'
    return response

@app.exception_handler(PhaseError)
async def phase_error(request,e):
    if e.code in ADMISSION_CODES:
        return JSONResponse({'error':e.code,'message':MESSAGES[e.code],'category':'ADMISSION_REJECTED'},413 if e.code=='FILE_TOO_LARGE' else 415 if e.code in {'FILE_TYPE_MISMATCH','UNSUPPORTED_FILE_TYPE'} else 422)
    code=404 if e.code=='NOT_FOUND' else 409 if e.code in {'REVISION_CONFLICT','IDEMPOTENCY_CONFLICT'} else 422
    return JSONResponse({'error':e.code},code)

@app.exception_handler(sqlite3.OperationalError)
async def db_busy(request,e):return JSONResponse({'error':'DATABASE_BUSY_RETRY_LATER'},503,headers={'Retry-After':'3'})

class DecisionBody(BaseModel):
    request_id:str=Field(min_length=1,max_length=100)
    actor:str=Field(min_length=1,max_length=100)
    candidate_id:str|None=None
    selected_form:Literal['raw','normalized','manual','unresolved']
    manual_text:str|None=Field(default=None,max_length=2000)
    reason:str=Field(min_length=1,max_length=500)
    expected_revision:int=Field(ge=0)

@app.get('/',response_class=HTMLResponse)
def home(request:Request):
    return RedirectResponse('/app',303)

@app.post('/documents',status_code=202)
async def upload(file:UploadFile=File(...),mode:Literal['LIVE','REPLAY']=Form('LIVE'),cloud_enabled:bool=Form(True),idempotency_key:str|None=Header(None)):
    s=Store();data=await read_upload(file,s.root)
    return s.upload(data,file.filename or '',mode,cloud_enabled,idempotency_key)

@app.get('/documents')
def documents():return Store().all('SELECT * FROM jobs ORDER BY created DESC LIMIT 200')

@app.get('/documents/{document_id}')
def document(document_id:str):
    s=Store();job=s.job(document_id)
    job['pages']=s.all('SELECT * FROM page_stages WHERE job_id=? ORDER BY page_index',(document_id,))
    job['reviews']=s.all('SELECT id,page_index,cell_id,status,revision FROM review_items WHERE job_id=? ORDER BY page_index,cell_id',(document_id,))
    return job

@app.get('/documents/{document_id}/result')
def result(document_id:str):
    s=Store();j=s.job(document_id)
    if j['processing_contract']=='V3':return JSONResponse({'error':'RESULT_CONTRACT_V3_REQUIRED','location':f'/v3/documents/{document_id}/result'},409)
    return s.read(j['result_ref']) if j['result_ref'] else JSONResponse({'status':j['status'],'rows':[]},202)

@app.get('/jobs/{document_id}',response_class=HTMLResponse)
def job_page(request:Request,document_id:str):
    return templates.TemplateResponse(request=request,name='job.html',context={'job':document(document_id)})

@app.get('/reviews')
def reviews(status:str|None=None,target_kind:str|None=None,offset:int=0,limit:int=100):
    s=Store()
    where=[];args=[]
    for key,val in [('status',status),('target_kind',target_kind)]:
        if val:where.append(key+'=?');args.append(val)
    return s.all('SELECT id,job_id,page_index,cell_id,target_kind,target_id,review_requirement,status,revision FROM review_items'+(' WHERE '+' AND '.join(where) if where else '')+' ORDER BY rowid LIMIT ? OFFSET ?',args+[max(1,min(500,limit)),max(0,offset)])

@app.get('/review-queue',response_class=HTMLResponse)
def review_queue(request:Request,offset:int=0,target_kind:str|None=None):
    return templates.TemplateResponse(request=request,name='reviews.html',context={'reviews':reviews(target_kind=target_kind,offset=offset),'offset':max(0,offset),'target_kind':target_kind or ''})

@app.get('/reviews/{review_id}')
def get_review(review_id:str):return review(Store(),review_id)

@app.get('/review/{review_id}',response_class=HTMLResponse)
def review_page(request:Request,review_id:str):
    r=review(Store(),review_id)
    if r['target_kind']=='LINE':return templates.TemplateResponse(request=request,name='review_line.html',context={'review':r,'line':r['evidence']['line'],'evidence':r['evidence']})
    return templates.TemplateResponse(request=request,name='review.html',context={'review':r,'cell':r['evidence']['cell'],'evidence':r['evidence']})

@app.post('/reviews/{review_id}/decision')
def decision(review_id:str,body:DecisionBody,request:Request):
    data=body.model_dump()
    if getattr(request.state,'principal',None):data['actor']=request.state.principal['user']
    return decide(Store(),review_id,data)

@app.get('/artifacts/{document_id}/{page_index}/{kind}/{cell_id}')
def artifact(document_id:str,page_index:int,kind:str,cell_id:str):
    s=Store();s.job(document_id)
    p=s.one('SELECT artifact_ref FROM page_stages WHERE job_id=? AND page_index=?',(document_id,page_index))
    if not p or not p['artifact_ref']:raise PhaseError('NOT_FOUND')
    if kind=='page':ref=p['artifact_ref']+'/canonical_page.png'
    elif kind=='crop':
        r=s.one('SELECT id,target_kind,evidence FROM review_items WHERE job_id=? AND page_index=? AND target_id=?',(document_id,page_index,cell_id))
        if not r:raise PhaseError('NOT_FOUND')
        ev=json.loads(r['evidence'])
        if not ev['line' if r['target_kind']=='LINE' else 'cell'].get('bbox'):raise PhaseError('POSITION_UNKNOWN')
        ref=p['artifact_ref']+'/crops/'+cell_id+'.png'
        if r['target_kind']=='LINE':
            from PIL import Image
            import io
            box=json.loads(r['evidence'])['line'].get('bbox')
            if not box:raise PhaseError('POSITION_UNKNOWN')
            with Image.open(s.resolve(p['artifact_ref']+'/canonical_page.png')) as image:
                out=io.BytesIO();image.crop(box).save(out,format='PNG')
            return Response(out.getvalue(),media_type='image/png')
    else:raise PhaseError('NOT_FOUND')
    path=s.resolve(ref)
    if not path.is_file():raise PhaseError('ARTIFACT_MISSING_OR_INVALID')
    return FileResponse(path,media_type='image/png')

@app.get('/api/budget')
def budget():return Store().stats()

from .views03 import router
app.include_router(router)

@app.post('/v3/documents',status_code=202)
async def upload_v3(file:UploadFile=File(...),mode:Literal['LIVE','REPLAY']=Form('LIVE'),cloud_enabled:bool=Form(True),idempotency_key:str|None=Header(None),ocr_provider:Literal['baidu','deepseek']|None=Form(None)):
    s=Store()
    return s.upload(await read_upload(file,s.root),file.filename or '',mode,cloud_enabled,idempotency_key,processing_contract='V3',ocr_provider=ocr_provider)

@app.get('/v3/documents/{document_id}/result')
def result_v3(document_id:str):
    s=Store();job=s.job(document_id)
    if job['processing_contract']!='V3':return JSONResponse({'error':'RESULT_CONTRACT_V2_REQUIRED'},409)
    return s.read(job['result_ref']) if job['result_ref'] else JSONResponse({'schema_version':'ocr-result-v3','status':job['status'],'pages':[]},202)

class RequirementBody(BaseModel):
    request_id:str=Field(min_length=1,max_length=100)
    actor:str=Field(min_length=1,max_length=100)
    reason:str=Field(min_length=1,max_length=500)
    expected_revision:int=Field(ge=0)

@app.post('/reviews/{review_id}/require')
def require_review(review_id:str,body:RequirementBody,request:Request):
    from .workflow_v3 import require_line
    data=body.model_dump()
    if getattr(request.state,'principal',None):data['actor']=request.state.principal['user']
    return require_line(Store(),review_id,data)
