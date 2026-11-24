"""Infer text flow from ink rows/columns, independently for each region."""
import cv2
import numpy as np
from PIL import Image

def bands(values,bridge=0):
    runs=[]; start=None
    for i,v in enumerate(list(values)+[False]):
        if v and start is None: start=i
        elif not v and start is not None: runs.append([start,i]); start=None
    merged=[]
    for a,b in runs:
        if merged and a-merged[-1][1]<=bridge: merged[-1][1]=b
        else: merged.append([a,b])
    return merged

def infer_direction(mask,language='ja'):
    """Return heuristic direction and reading-order line boxes, not a probability.

    Strong inter-column or inter-row gutters beat the enclosing rectangle's aspect.
    Ambiguous grids use a disclosed fallback and remain reviewable.
    """
    ink=(np.asarray(mask)>96).astype(np.uint8)
    points=cv2.findNonZero(ink)
    if points is None: return {'orientation':'horizontal','orientation_method':'empty','direction_uncertain':True,'source_lines':[]}
    x,y,w,h=cv2.boundingRect(points)
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,8)
    sizes=[max(s[2],s[3]) for s in stats[1:] if s[4]>=8 and min(s[2],s[3])>=3]
    glyph=max(8,min(100,float(np.percentile(sizes,65)))) if sizes else 20
    bridge=max(1,round(glyph*.16))
    columns=bands(ink.any(axis=0),bridge); rows=bands(ink.any(axis=1),bridge)
    def gutter(parts):
        gaps=[b[0]-a[1] for a,b in zip(parts,parts[1:])]
        return float(np.percentile(gaps,65))/glyph if gaps else 0.
    gx,gy=gutter(columns),gutter(rows)
    # Closely set manga often has only a 3-5 px column gutter; count complete
    # narrow ink columns before comparing a single unusually large row gap.
    fine_columns=bands(ink.any(axis=0),max(1,round(glyph*.06)))
    fine_rows=bands(ink.any(axis=1),max(1,round(glyph*.06)))
    substantial_columns=[(a,b) for a,b in fine_columns if b-a>=glyph*.35]
    substantial_rows=[(a,b) for a,b in fine_rows if b-a>=glyph*.35]
    narrow_columns=(1<len(substantial_columns)<=5 and h>w*1.6 and
                    all(b-a<glyph*1.7 for a,b in substantial_columns) and len(substantial_rows)<=3)
    uncertain=False; method='ink-gutters'
    if language=='en': orientation='horizontal'; method='latin-lines'
    elif narrow_columns:
        orientation='vertical'; method='compact-ink-columns'; columns=[list(v) for v in substantial_columns]
    elif len(columns)==1 and h>w*1.7 and w<glyph*1.8:
        orientation='vertical'; method='single-column'
    elif len(rows)==1 and w>h*1.4 and h<glyph*1.8:
        orientation='horizontal'; method='single-row'
    elif gx>.14 and gx>gy*1.7:
        orientation='vertical'
    elif gy>.3 and gy>gx*1.7:
        orientation='horizontal'
    elif gy>.18 and gx<.05 and w>glyph*3 and h<w*1.3 and len(rows)>=2:
        # Handwritten notes: clear row gaps, no column gutter at all, several glyphs wide.
        orientation='horizontal'; method='row-gaps-no-columns'; uncertain=True  # whole-block OCR reads handwriting better
    else:
        orientation='vertical' if h>w*1.3 or (language=='ja' and h>=w*.85) else 'horizontal'
        uncertain=True; method='ambiguous-aspect-fallback'
    if orientation=='vertical' and method=='ink-gutters' and w>h*1.3 and len(rows)<=3:
        # Gaps between handwritten characters look like column gutters, but real
        # columns are about one glyph wide; one or two wide "columns" are rows.
        wide=bands(ink.any(axis=0),bridge*3)
        if wide and np.median([b-a for a,b in wide])>glyph*2.2:
            orientation='horizontal'; method='wide-bands-not-columns'; uncertain=True
    if (language=='ja' and orientation=='horizontal' and method=='ink-gutters' and len(columns)==1
            and len(rows)>=2 and h>w*1.2):
        # One vertical column whose glyphs and ellipsis dots stand apart (男… / おクッソ……)
        # looks like stacked rows; real rows are wider than tall, these are single glyphs.
        spans=[np.flatnonzero(ink[a:b].any(axis=0)) for a,b in rows]
        if all(len(s) and s[-1]-s[0]+1<=1.3*(b-a) for s,(a,b) in zip(spans,rows)):
            orientation='vertical';method='stacked-glyph-column'
    lines=[]
    if orientation=='vertical':
        # Small ruby (furigana) to the right belongs to the preceding full-size column.
        max_width=max((b-a for a,b in columns),default=1)
        merged=[]
        for a,b in columns:
            if language=='ja' and merged and b-a<max_width*.5: merged[-1][1]=b
            else: merged.append([a,b])
        for a,b in reversed(merged):
            ys=np.flatnonzero(ink[:,a:b].any(axis=1))
            if len(ys): lines.append([max(0,a-2),max(0,int(ys[0])-2),min(mask.width,b+2),min(mask.height,int(ys[-1])+3)])
    else:
        for a,b in rows:
            xs=np.flatnonzero(ink[a:b].any(axis=0))
            if len(xs): lines.append([max(0,int(xs[0])-2),max(0,a-2),min(mask.width,int(xs[-1])+3),min(mask.height,b+2)])
    return {'orientation':orientation,'orientation_method':method,'direction_uncertain':uncertain,
            'source_lines':lines,'gutter_x':round(gx,3),'gutter_y':round(gy,3)}

def set_direction(region,mask,language):
    info=infer_direction(mask.crop(region['box']),language)
    region.update(info)
    override=region.get('direction_override','auto')
    if override in ('vertical','horizontal'):
        region['render_direction']=override; region['direction_uncertain']=False
    else: region['render_direction']=info['orientation']
    return region['render_direction']
