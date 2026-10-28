"""Select actual transport bytes; canonical pixels remain an independent artifact."""
import copy
from pathlib import Path
from PIL import Image
from .common import digest
from .baidu import request_spec
from .line_grouping import valid_box


def select(path, out, meta, mode='LIVE', cache_dir=None):
    path,out=Path(path),Path(out)
    canonical=out/'canonical_page.png'
    ref=path if meta['kind']=='image' else canonical
    source='ORIGINAL_PASSTHROUGH' if meta['kind']=='image' else 'PDF_RENDER_PNG'
    strategy='IMAGE_FILE' if meta['kind']=='image' else 'RENDERED_IMAGE'
    data=ref.read_bytes()
    # Explicit historical replay uses the historical bytes and identity, never aliases JPEG to PNG.
    if mode=='REPLAY' and cache_dir and meta['kind']=='image':
        _,newkey=request_spec(data,strategy,meta['page_index'])
        olddata=canonical.read_bytes();_,oldkey=request_spec(olddata,strategy,meta['page_index'])
        if not (Path(cache_dir)/newkey/'request.json').exists() and (Path(cache_dir)/oldkey/'request.json').exists():
            ref=canonical;data=olddata;source='LEGACY_CANONICAL_REPLAY'
    with Image.open(ref) as im:
        width,height=im.size;fmt=im.format;orientation=im.getexif().get(274,1)
        # Pillow does not auto-apply EXIF when decoding. Compare the raw RGB matrix with canonical.
        with Image.open(canonical) as ci:
            same=im.size==ci.size and orientation==1 and im.convert('RGB').tobytes()==ci.convert('RGB').tobytes()
    return data,dict(original_artifact_ref=str(path.resolve()),original_size=path.stat().st_size,
        original_hash=meta['input_hash'],canonical_artifact_ref=str(canonical.resolve()),
        canonical_hash=digest(canonical.read_bytes()),canonical_width=meta['width'],canonical_height=meta['height'],
        provider_input_ref=str(ref.resolve()),provider_input_hash=digest(data),provider_input_format=fmt,
        provider_input_width=width,provider_input_height=height,provider_input_size=len(data),
        provider_payload_size=None,provider_representation_source=source,provider_exif_orientation=orientation,
        canonical_exif_applied=meta.get('exif_orientation',1),pixel_matrix_equal=same,
        coordinate_status='VALID' if same else 'UNKNOWN',
        provider_to_canonical_transform='identity' if same else None,
        coordinate_transform='identity' if same else None,input_strategy=strategy)


def map_geometry(items, representation, coordinate_status, direction):
    """Preserve provider boxes separately, quarantine uncertain/out-of-bounds geometry."""
    items=copy.deepcopy(items);geometry=[]
    known=representation['coordinate_status']=='VALID' and coordinate_status=='CANONICAL_INPUT_PIXELS' and direction in (None,-1,0)
    w,h=representation['provider_input_width'],representation['provider_input_height']
    def record(atom):
        box=atom.get('provider_bbox',atom.get('bbox'));bounded=valid_box(box) and 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
        status='INVALID' if not bounded or atom.get('geometry_status')=='INVALID' else 'VALID' if known else 'UNKNOWN'
        geometry.append(dict(fragment_id=atom['id'],provider_bbox=box,canonical_bbox=box if status=='VALID' else None,geometry_status=status))
        if status!='VALID':
            atom['bbox']=None;atom['geometry_status']=status
            atom['risk_reasons']=sorted(set(atom.get('risk_reasons',[])+['INVALID_PROVIDER_BBOX' if status=='INVALID' else 'COORDINATE_SPACE_UNKNOWN']))
        return status
    for item in items:
        child_status=[record(c) for c in item.get('characters',[])]
        status=record(item)
        if 'INVALID' in child_status and status!='INVALID':
            item.update(bbox=None,geometry_status='INVALID',risk_reasons=sorted(set(item.get('risk_reasons',[])+['INVALID_PROVIDER_BBOX'])))
            geometry[-1].update(canonical_bbox=None,geometry_status='INVALID')
    return items,geometry,'CANONICAL_INPUT_PIXELS' if known else 'COORDINATE_SPACE_UNKNOWN'
