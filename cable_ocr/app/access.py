"""Temporary test access control, not a registration/RBAC system."""
import base64,hashlib,hmac,json,os,time
from pathlib import Path
from dotenv import dotenv_values


def required():return os.environ.get('CABLE_REQUIRE_AUTH','0')=='1'
def settings():
    if not required():return None
    p=Path(os.environ.get('CABLE_ACCESS_ENV','data/private/access.env'))
    v=dotenv_values(p) if p.is_file() else {}
    if not all(v.get(k) for k in ['TEST_UI_USERNAME','TEST_UI_PASSWORD','ADMIN_USERNAME','ADMIN_PASSWORD']):return {}
    return v

def check(user,password,v):
    if not v:return None
    for role,prefix in [('admin','ADMIN'),('user','TEST_UI')]:
        if hmac.compare_digest(user.encode(),v[prefix+'_USERNAME'].encode()) and hmac.compare_digest(password.encode(),v[prefix+'_PASSWORD'].encode()):return role
    return None

def key(v):return hashlib.sha256((v['ADMIN_PASSWORD']+'\0'+v['TEST_UI_PASSWORD']).encode()).digest()
def token(role,user,v):
    payload=base64.urlsafe_b64encode(json.dumps({'role':role,'user':user,'expires':int(time.time())+3600}).encode()).decode()
    return payload+'.'+hmac.new(key(v),payload.encode(),hashlib.sha256).hexdigest()
def principal(request,v):
    auth=request.headers.get('authorization','')
    if auth.startswith('Basic '):
        try:
            user,password=base64.b64decode(auth[6:],validate=True).decode().split(':',1)
            role=check(user,password,v)
            if role:return {'role':role,'user':user}
        except (ValueError,UnicodeError):pass
    cookie=request.cookies.get('cable_test_session','')
    try:
        body,sig=cookie.split('.')
        if not hmac.compare_digest(sig,hmac.new(key(v),body.encode(),hashlib.sha256).hexdigest()):return None
        p=json.loads(base64.urlsafe_b64decode(body))
        if p['expires']>time.time() and p['role'] in {'admin','user'}:return p
    except (ValueError,KeyError,UnicodeError):pass
    return None

def user_area(path):return path=='/' or path=='/app' or path.startswith('/app/')
