"""Product input contract; rejects before creating jobs. No input rewriting."""
import io,json,time,fcntl,os
from pathlib import Path
from PIL import Image
from .common import PhaseError
MAX_FILE_BYTES=10*1024*1024
MAX_IMAGE_PIXELS=40_000_000
MESSAGES={
 'FILE_TOO_LARGE':'文件超过 10 MiB，请上传不超过 10 MiB 的原始文件。',
 'IMAGE_DIMENSION_EXCEEDED':'图片尺寸超过当前支持范围。',
 'FILE_TYPE_MISMATCH':'文件扩展名与实际格式不一致，请重新导出或上传正确格式文件。',
 'UNSUPPORTED_FILE_TYPE':'仅支持 PDF、PNG、JPEG/JPG 文件。',
 'INVALID_FILE':'文件内容无效或已损坏，请检查原始文件。',
 'PROVIDER_PAYLOAD_TOO_LARGE':'当前文件虽然符合上传限制，但生成的 OCR 请求超过云识别服务限制，暂时无法处理。'}
ADMISSION_CODES=set(MESSAGES)
def validate_dimensions(width,height):
 if width*height>MAX_IMAGE_PIXELS:raise PhaseError('IMAGE_DIMENSION_EXCEEDED',{'width':width,'height':height,'pixel_limit':MAX_IMAGE_PIXELS})
def validate(data,name):
 ext=Path(name).suffix.lower();detected='PDF' if data.startswith(b'%PDF-') else 'PNG' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'JPEG' if data.startswith(b'\xff\xd8\xff') else 'UNKNOWN'
 details={'file_size':len(data),'declared_extension':ext[:16],'detected_type':detected}
 def fail(code):raise PhaseError(code,details)
 if len(data)>MAX_FILE_BYTES:fail('FILE_TOO_LARGE')
 if ext not in {'.pdf','.png','.jpg','.jpeg'}:fail('UNSUPPORTED_FILE_TYPE')
 if not data or detected=='UNKNOWN':fail('INVALID_FILE')
 if detected!={'.pdf':'PDF','.png':'PNG','.jpg':'JPEG','.jpeg':'JPEG'}[ext]:fail('FILE_TYPE_MISMATCH')
 if detected in {'PNG','JPEG'}:
  try:
   with Image.open(io.BytesIO(data)) as im:
    details.update(width=im.width,height=im.height,detected_type=im.format)
    if im.format!=detected:fail('FILE_TYPE_MISMATCH')
    validate_dimensions(im.width,im.height);im.verify()
   with Image.open(io.BytesIO(data)) as decoded:decoded.load()
  except PhaseError as e:e.details={**details,**e.details};raise
  except Exception:fail('INVALID_FILE')
 else:
  import pypdfium2 as pdfium
  import pypdfium2.raw as raw
  try:
   with pdfium.PdfDocument(data) as doc:
    if len(doc)<1:fail('INVALID_FILE')
  except PhaseError:raise
  except Exception as e:
   # Encryption remains an explicit ingestion boundary, never an OCR request.
   if getattr(e,'err_code',None)!=raw.FPDF_ERR_PASSWORD:fail('INVALID_FILE')
 return details

def log_rejection(root,code,details=None):
 allowed={'file_size','declared_extension','detected_type','width','height','pixel_limit','form_bytes','page_index','job_id','request_bytes'}
 record={'timestamp':time.time(),'category':'ADMISSION_REJECTED','reason':code,**{k:v for k,v in (details or {}).items() if k in allowed}}
 path=Path(root)/'admission_rejections.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 with os.fdopen(fd,'a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 return record

async def read_upload(file,root):
 if file.size is not None and file.size>MAX_FILE_BYTES:
  details={'file_size':file.size,'declared_extension':Path(file.filename or '').suffix.lower()[:16]}
  log_rejection(root,'FILE_TOO_LARGE',details);raise PhaseError('FILE_TOO_LARGE',details)
 return await file.read(MAX_FILE_BYTES+1)

def rejection_summary(root):
 import collections
 counts=collections.Counter();rows=[];path=Path(root)/'admission_rejections.jsonl'
 if path.exists():
  with path.open() as f:
   for line in f:
    try:r=json.loads(line)
    except ValueError:continue
    counts[r['reason']]+=1;rows.append(r)
    if len(rows)>1000:rows.pop(0)
 return {'count':sum(counts.values()),'reasons':dict(counts),'recent':list(reversed(rows))}
