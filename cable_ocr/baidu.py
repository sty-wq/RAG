"""Extracted from connect_example.py's endpoint/OAuth/form contract.

Only page recognition; no local OCR or provider hierarchy. Responses are durable
before interpretation. REPLAY never reads credentials or opens a connection.
"""
import base64
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode
import requests
from dotenv import dotenv_values
from .common import PhaseError, digest, save_bytes, save_json, stable_hash

ENDPOINT = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
TOKEN_ENDPOINT = "https://aip.baidubce.com/oauth/2.0/token"
PROFILE = "accurate_with_location"
PARSER_VERSION = "baidu-text-preserving-geometry-v2"
OPTIONS = {"language_type":"auto_detect", "eng_granularity":"letter",
           "recognize_granularity":"small", "detect_direction":"false",
           "vertexes_location":"false", "paragraph":"false", "probability":"true",
           "char_probability":"true", "multidirectional_recognize":"false"}


def read_credentials(path):
    values=dotenv_values(path) if Path(path).is_file() else {}
    pair=[values.get(k) for k in ("API_KEY","SECRET_KEY")]
    if not all(isinstance(v,str) and v.strip() for v in pair):
        raise PhaseError("MISSING_CREDENTIALS")
    return pair


def request_spec(input_bytes, strategy, page_index, options=None):
    if strategy not in {"PDF_DIRECT","RENDERED_IMAGE","IMAGE_FILE"} or page_index<0:
        raise PhaseError("INVALID_INPUT_STRATEGY")
    spec={"input_hash":digest(input_bytes),"provider":"baidu","endpoint":ENDPOINT,"profile":PROFILE,
          "input_strategy":strategy,"page_index":page_index,"options_version":"phase01-v1",
          "preprocessing_version":"canonical-pillow-pdfium-v1",
          "options":dict(OPTIONS if options is None else options)}
    return spec,stable_hash(spec)


def request_payload(input_bytes,strategy,page_index=0):
    encoded=base64.b64encode(input_bytes).decode('ascii')
    form=dict(OPTIONS)
    if strategy=="PDF_DIRECT":form.update(pdf_file=encoded,pdf_file_num=str(page_index+1))
    else:form['image']=encoded
    payload=urlencode(form).encode('ascii')
    if len(payload)>10*1024*1024:raise PhaseError('PROVIDER_PAYLOAD_TOO_LARGE',{'form_bytes':len(payload)})
    return payload


def location(value):
    """Unusable geometry is nullable evidence, not a page-level text failure."""
    if not isinstance(value,dict):return None
    vals=[value.get(k) for k in ("left","top","width","height")]
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vals):return None
    x,y,w,h=vals
    if x<0 or y<0 or w<=0 or h<=0 or not math.isfinite(x+w) or not math.isfinite(y+h) or x+w<=x or y+h<=y:return None
    return [x,y,x+w,y+h]


def parse_response(raw):
    if not isinstance(raw,dict): raise PhaseError("MALFORMED_BAIDU_RESPONSE")
    if "error_code" in raw:
        raise PhaseError("BAIDU_API_ERROR",{"provider_error_code":raw["error_code"],"log_id":raw.get("log_id")})
    words=raw.get("words_result")
    if not isinstance(words,list): raise PhaseError("MALFORMED_BAIDU_RESPONSE",{"field":"words_result"})
    if raw.get("words_result_num",len(words))!=len(words):
        raise PhaseError("MALFORMED_BAIDU_RESPONSE",{"field":"words_result_num"})
    items=[]
    for i,v in enumerate(words):
        if not isinstance(v,dict) or not isinstance(v.get("words"),str):
            raise PhaseError("MALFORMED_BAIDU_RESPONSE",{"field":"words"})
        chars=v.get("chars",[])
        bad_character_metadata=not isinstance(chars,list)
        if bad_character_metadata:chars=[]
        characters=[]
        for j,c in enumerate(chars):
            if not isinstance(c,dict) or not isinstance(c.get("char"),str):
                bad_character_metadata=True;continue
            char={'id':f"w{i}.c{j}","char":c['char'],"bbox":location(c.get('location')),
                  "confidence":c.get('char_prob')}
            if char['bbox'] is None:char.update(geometry_status='INVALID',risk_reasons=['INVALID_PROVIDER_BBOX'])
            characters.append(char)
        box=location(v.get('location'))
        invalid=bad_character_metadata or box is None or any(c['bbox'] is None for c in characters)
        item={"id":f"w{i}","text":v['words'],"bbox":None if invalid else box,
              "confidence":v.get('probability'),"characters":characters}
        if invalid:item.update(provider_bbox=box,geometry_status='INVALID',risk_reasons=['INVALID_PROVIDER_BBOX'],provider_ordinal=i)
        if bad_character_metadata:item['risk_reasons'].append('INVALID_PROVIDER_CHARACTER_METADATA')
        items.append(item)
    return items


class BaiduCloudOCRProvider:
    def __init__(self, env_path, cache_dir, session=None):
        self.env_path=Path(env_path); self.cache_dir=Path(cache_dir)
        self.session=session  # lazy: REPLAY needs neither requests.Session nor .env
        self._token=None; self._expires=0

    def _access_token(self):
        if self._token and time.monotonic()<self._expires: return self._token
        key,secret=read_credentials(self.env_path)
        if self.session is None: self.session=requests.Session()
        try:
            r=self.session.post(TOKEN_ENDPOINT,params={"grant_type":"client_credentials","client_id":key,"client_secret":secret},timeout=(10,30))
            raw=r.json()
        except PhaseError:
            raise
        except Exception as e:
            raise PhaseError("AUTH_REQUEST_FAILED",{"exception_type":type(e).__name__}) from None
        if r.status_code!=200 or not isinstance(raw,dict) or not isinstance(raw.get('access_token'),str):
            raise PhaseError("AUTH_FAILED")
        self._token=raw['access_token']
        self._expires=time.monotonic()+max(0,int(raw.get('expires_in',0))-60)
        return self._token

    def recognize_page(self, input_bytes, strategy, page_index=0, mode="REPLAY"):
        if mode not in {"LIVE","REPLAY"}: raise PhaseError("INVALID_EXECUTION_MODE")
        spec,key=request_spec(input_bytes,strategy,page_index)
        folder=self.cache_dir/key; manifest_path=folder/'request.json'; raw_path=folder/'response.bin'
        cache_hit=manifest_path.exists()
        if cache_hit:
            try:
                record=json.loads(manifest_path.read_text()); body=raw_path.read_bytes()
            except Exception:
                raise PhaseError("CACHE_INCOMPLETE") from None
            if record.get('request_key')!=key or record.get('spec')!=spec or record.get('raw_sha256')!=digest(body):
                raise PhaseError("CACHE_INTEGRITY_ERROR")
        elif mode=="REPLAY":
            raise PhaseError("REPLAY_CACHE_MISS",{"request_key":key})
        else:
            payload=request_payload(input_bytes,strategy,page_index)
            token=self._access_token()
            start=time.monotonic()
            # One attempt only: no hidden paid retries. HTTP errors are sanitized.
            try:
                r=self.session.post(ENDPOINT,params={'access_token':token},data=payload,
                    headers={'Content-Type':'application/x-www-form-urlencoded','Accept':'application/json'},timeout=(10,90))
                body=r.content
            except PhaseError:
                raise
            except Exception as e:
                save_json(folder/'attempt_error.json',{'request_key':key,'spec':spec,'status':'NETWORK_OUTCOME_UNKNOWN',
                          'exception_type':type(e).__name__,'elapsed_ms':(time.monotonic()-start)*1000,
                          'billing_status':'UNKNOWN'})
                raise PhaseError("OCR_NETWORK_OUTCOME_UNKNOWN") from None
            record={'request_key':key,'spec':spec,'raw_sha256':digest(body),'http_status':r.status_code,
                    'input_bytes':len(input_bytes),'form_bytes':len(payload),'pdf_page_number':page_index+1 if strategy=='PDF_DIRECT' else None,
                    'headers':{'Content-Type':'application/x-www-form-urlencoded','Accept':'application/json'},
                    'recorded_live_latency_ms':(time.monotonic()-start)*1000,'logical_calls':1,'http_attempts':1,
                    'billing_status':'NOT_RECONCILED_WITH_BILL','auth':'OAuth client_credentials; token omitted'}
            save_bytes(raw_path,body)
            # The manifest commits the raw artifact. No token response is stored.
            save_json(manifest_path,record)
        try: raw=json.loads(body)
        except Exception: raise PhaseError("MALFORMED_BAIDU_RESPONSE",{'raw_ref':str(raw_path)}) from None
        if not (folder/'response.json').exists():save_json(folder/'response.json',raw)
        if record['http_status']!=200: raise PhaseError("OCR_HTTP_ERROR",{'status':record['http_status'],'raw_ref':str(raw_path)})
        items=parse_response(raw)
        return {'provider':'baidu','profile':PROFILE,'input_strategy':strategy,'page_index':page_index,
                'items':items,'direction':raw.get('direction'),'raw_response_ref':str(folder/'response.json'),
                'raw_bytes_ref':str(raw_path),'request_metadata':record,'request_key':key,
                'execution_mode':mode,'cache_hit':cache_hit,'new_ocr_http_calls':0 if cache_hit else 1,
                'coordinate_status':'COORDINATE_SPACE_UNKNOWN' if strategy=='PDF_DIRECT' else 'CANONICAL_INPUT_PIXELS'}
