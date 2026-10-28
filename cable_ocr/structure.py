"""Axis-aligned, complete ruled grids only; no OCR input or model."""
import numpy as np
from PIL import Image,ImageFilter


def centers(indices):
    groups=[]
    for i in indices:
        if not groups or i > groups[-1][-1]+1:
            groups.append([int(i)])
        else:
            groups[-1].append(int(i))
    return [int(round(sum(g)/len(g))) for g in groups]


def threshold(gray):
    histogram=np.bincount(gray.ravel(),minlength=256)
    counts=histogram.cumsum(); weights=(histogram*np.arange(256)).cumsum()
    denominator=counts*(gray.size-counts)
    score=np.zeros(256,dtype=float)
    np.divide((float(weights[-1])*counts-weights*float(gray.size))**2,denominator,out=score,where=denominator>0)
    return gray<=int(score.argmax())


def long_runs(mask,length,vertical=False):
    """Retain straight dark runs; no general CV framework or model."""
    view=mask.T if vertical else mask
    edges=np.diff(np.pad(view.astype(np.int8),((0,0),(1,1))),axis=1)
    rows,starts=np.where(edges==1);_,ends=np.where(edges==-1)
    output=np.zeros_like(view,dtype=bool)
    for row,start,end in zip(rows,starts,ends):
        if end-start>=length:output[row,start:end]=True
    return output.T if vertical else output


def line_regions(mask):
    """Bounded 4-connected traversal of the sparse, dilated line network."""
    network=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3)))>0
    if network.sum()>2_000_000 or network.mean()>.25:return []
    visited=np.zeros_like(network);h,w=network.shape;regions=[]
    yy,xx=np.nonzero(network)
    for y,x in zip(yy.tolist(),xx.tolist()):
        if visited[y,x]:continue
        stack=[(y,x)];visited[y,x]=True;x0=x1=x;y0=y1=y
        while stack:
            cy,cx=stack.pop();x0=min(x0,cx);x1=max(x1,cx);y0=min(y0,cy);y1=max(y1,cy)
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<h and 0<=nx<w and network[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx]=True;stack.append((ny,nx))
        regions.append((x0,y0,x1-x0+1,y1-y0+1))
    return regions


def recover(image):
    gray=np.asarray(image.convert("L"))
    binary=threshold(gray)
    h,w=binary.shape
    horizontal=long_runs(binary,max(20,w//30))
    vertical=long_runs(binary,max(20,h//30),vertical=True)
    # Connected line network, not OCR words, determines table regions.
    regions=[]
    for x,y,rw,rh in line_regions(horizontal|vertical):
        if rw>=w*.15 and rh>=max(30,h*.025): regions.append((x,y,rw,rh))
    regions.sort(key=lambda r:(r[1],r[0]))
    tables=[]; issues=[]
    for n,(x,y,rw,rh) in enumerate(regions):
        ys=[y+z for z in centers(np.flatnonzero((horizontal[y:y+rh,x:x+rw]>0).sum(axis=1)>.8*rw))]
        xs=[x+z for z in centers(np.flatnonzero((vertical[y:y+rh,x:x+rw]>0).sum(axis=0)>.8*rh))]
        if len(xs)<3 or len(ys)<3:
            issues.append({"code":"UNSUPPORTED_STRUCTURE","region":[x,y,x+rw,y+rh]}); continue
        if len(xs)>31 or len(ys)>301 or min(np.diff(xs))<12 or min(np.diff(ys))<12:
            issues.append({"code":"STRUCTURE_REVIEW_REQUIRED","reason":"grid_spacing_or_count"}); continue
        broken=False
        for yy in ys:
            for a,b in zip(xs,xs[1:]):
                if (horizontal[max(0,yy-2):yy+3,a:b]>0).any(axis=0).mean()<.9: broken=True
        for xx in xs:
            for a,b in zip(ys,ys[1:]):
                if (vertical[a:b,max(0,xx-2):xx+3]>0).any(axis=1).mean()<.9: broken=True
        # Partial dividers reveal merged/broken grids; do not silently merge bands.
        # A repeated glyph stem is not a partial grid divider. Require a run
        # spanning most of the smallest inferred cell before checking partial lines.
        partial_h=long_runs(binary,max(20,int(min(np.diff(xs))*.65)))
        partial_v=long_runs(binary,max(20,int(min(np.diff(ys))*.65)),vertical=True)
        hy=centers(np.flatnonzero(partial_h[y:y+rh,x:x+rw].sum(axis=1)>max(20,min(np.diff(xs))*.65)))
        vx=centers(np.flatnonzero(partial_v[y:y+rh,x:x+rw].sum(axis=0)>max(20,min(np.diff(ys))*.65)))
        if any(min(abs(y+i-j) for j in ys)>4 for i in hy): broken=True
        if any(min(abs(x+i-j) for j in xs)>4 for i in vx): broken=True
        if broken:
            issues.append({"code":"STRUCTURE_REVIEW_REQUIRED","reason":"incomplete_or_merged_grid","region":[x,y,x+rw,y+rh]}); continue
        tid=f"table-{n}"
        rows=[]
        for ri,(y0,y1) in enumerate(zip(ys,ys[1:])):
            cells=[{"id":f"{tid}-r{ri}-c{ci}","row":ri,"column":ci,"bbox":[x0,y0,x1,y1],
                    "row_span":1,"column_span":1,"crop_ref":f"crops/{tid}-r{ri}-c{ci}.png",
                    "candidates":[],"decision":None,"status":"UNRESOLVED"}
                   for ci,(x0,x1) in enumerate(zip(xs,xs[1:]))]
            rows.append({"id":f"{tid}-r{ri}","row":ri,"role":"header" if ri==0 else "data","cells":cells})
        tables.append({"id":tid,"bbox":[xs[0],ys[0],xs[-1],ys[-1]],"row_count":len(rows),
                       "column_count":len(xs)-1,"structure_revision":1,"status":"PRE_PASSED","rows":rows})
    if not tables and not issues: issues=[{"code":"UNSUPPORTED_STRUCTURE","reason":"no_complete_ruled_table"}]
    return {"status":"STRUCTURE_REVIEW_REQUIRED" if issues else "PRE_PASSED","tables":tables,"issues":issues,
            "method":"numpy_pillow_straight_run_complete_grid_v1"}
