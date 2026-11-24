"""Group English line detections before OCR/translation; split OCR using real ink."""
import copy,re
import cv2
import numpy as np
from PIL import Image
from typesetting import BubbleMap

def glyph_height(mask):
    _,_,stats,_=cv2.connectedComponentsWithStats((np.asarray(mask)>96).astype(np.uint8),8)
    sizes=[min(w,h) for x,y,w,h,a in stats[1:] if a>=5 and min(w,h)>=3]
    return max(8,min(80,float(np.percentile(sizes,65)))) if sizes else 20

def merge_paragraphs(image,mask,regions,language):
    if language=='ja':
        from paragraph_geometry import merge_japanese
        return merge_japanese(image,mask,regions)
    if language!='en':return regions
    expanded=[]
    for index,r in enumerate(copy.deepcopy(regions)):
        x,y,X,Y=r['box'];crop=image.crop(r['box']);cmask=mask.crop(r['box'])
        columns=english_columns(crop,cmask)
        if len(columns)>1:
            for column,b in enumerate(columns):
                new=copy.deepcopy(r);new['box']=[x+b[0],y+b[1],x+b[2],y+b[3]]
                new['column_group']=index;new['column_index']=column;expanded.append(new)
        else:expanded.append(r)
    regions=expanded;bubbles=BubbleMap(image,regions);labels=[];glyph=glyph_height(mask)
    for r in regions:
        b=bubbles.find(r);labels.append(b['label'] if b else None)
    parents=list(range(len(regions)))
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    gray=np.asarray(image.convert('L'))
    for i,a in enumerate(regions):
        x,y,X,Y=a['box']
        for j in range(i):
            if a.get('column_group') is not None and a.get('column_group')==regions[j].get('column_group') and a.get('column_index')!=regions[j].get('column_index'):continue
            u,v,U,V=regions[j]['box']
            if labels[i] is not None and labels[j] is not None and labels[i]!=labels[j]:continue
            shared=labels[i] is not None and labels[i]==labels[j]
            # A huge effect adjacent to a small caption is not an extra line of
            # that caption. This prevents its mask from swallowing the prose.
            if not shared and max(Y-y,V-v)>min(Y-y,V-v)*6 and max(X-x,U-u)>min(X-x,U-u)*2.5:continue
            overlap=max(0,min(X,U)-max(x,u))/max(1,min(X-x,U-u))
            gap=max(y,v)-min(Y,V)
            same_line=max(0,min(Y,V)-max(y,v))/max(1,min(Y-y,V-v))>.55 and max(x,u)-min(X,U)<glyph*1.6
            adjacent=overlap>.45 and gap<glyph*1.15
            shared=shared and (gap<glyph*3 and overlap>.25 or same_line)
            if not (shared or adjacent or same_line):continue
            # A panel/balloon rule between lines is a hard boundary.
            if gap>0:
                strip=gray[min(Y,V):max(y,v),min(x,u):max(X,U)]
                if strip.size and ((strip<85).mean(1)>.8).any():continue
            if shared or adjacent or same_line:parents[root(i)]=root(j)
    groups={}
    for i,r in enumerate(regions):groups.setdefault(root(i),[]).append(r)
    result=[]
    for group in groups.values():
        r=group[0];boxes=[g['box'] for g in group]
        r['box']=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        r['paragraph_parts']=boxes;r['source_font_size']=round(glyph_height(mask.crop(r['box']))/0.8)
        result.append(r)
    result.sort(key=lambda r:(r['box'][1]//180,r['box'][0]))
    for i,r in enumerate(result):r['id']=i
    return result

def english_columns(crop,mask):
    """Separate side-by-side paragraphs before row OCR can interleave sentences."""
    ink=english_ink(crop,mask);h,w=ink.shape
    _,_,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    heights=[ch for x,y,cw,ch,a in stats[1:] if ch>=8 and cw>=4 and a>15]
    glyph=float(np.median(heights)) if heights else 20.
    if w<glyph*5 or h<glyph*4:return [(0,0,w,h)]
    columns=ink.sum(0);low=columns<=max(1,h*.025);runs=[];start=None
    for x,active in enumerate([*low,False]):
        if active and start is None:start=x
        elif not active and start is not None:
            if x-start>=max(4,glyph*.32) and start>=glyph*2.3 and w-x>=glyph*2.3:runs.append((start,x))
            start=None
    for a,b in sorted(runs,key=lambda r:r[1]-r[0],reverse=True):
        left=ink[:,:a];right=ink[:,b:];ly=np.flatnonzero(left.any(1));ry=np.flatnonzero(right.any(1))
        if not len(ly) or not len(ry):continue
        if min(ly[-1]-ly[0],ry[-1]-ry[0])<max(h*.28,glyph*3):continue
        if min(ly[-1],ry[-1])-max(ly[0],ry[0])<glyph*1.5:continue
        # A shared word space repeats the same baselines; independent columns
        # usually start/end on different rows or have a much wider gutter.
        if abs(int(ly[0])-int(ry[0]))<glyph*.7 and b-a<glyph*.65:continue
        result=[]
        for start,end in ((0,a),(b,w)):
            yy,xx=np.where(ink[:,start:end])
            result.append((max(start,start+int(xx.min())-2),max(0,int(yy.min())-2),min(end,start+int(xx.max())+3),min(h,int(yy.max())+3)))
        return result
    return [(0,0,w,h)]

def english_ink(crop,mask,expected_height=0):
    """Keep complete letter components, excluding rules/backgrounds before masking.

    Intersecting a background with CTD first creates artificial 'letters' from
    screen tones and balloon edges. Component filtering must precede that step.
    Black-filled, white-outlined narration uses its black interiors, too.
    """
    gray=np.asarray(crop.convert('L'));support=np.asarray(mask)>96
    candidates=[]
    for foreground in (gray<145,gray>185):
        n,labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
        selected=[]
        for i,(x,y,w,h,a) in enumerate(stats[1:],1):
            if a<3 or h<3 or a>gray.size*.38:continue
            if expected_height and (h<expected_height*.24 or a<expected_height**2*.015):continue
            if (y==0 and y+h==crop.height or x==0 and x+w==crop.width) and a>gray.size*.08:continue
            pixels=labels[y:y+h,x:x+w]==i
            fraction=float(support[y:y+h,x:x+w][pixels].mean())
            if fraction<.5:continue
            if h>crop.height*.72 and w<max(4,h*.12):continue
            if w>crop.width*.92 and h<4:continue
            selected.append((i,h,w,a))
        heights=[h for _,h,w,a in selected if h>=6 and w>=3]
        typical=float(np.median(heights)) if heights else 16.
        ids=[i for i,h,w,a in selected if h<=max(typical*2,12) and a<=max(typical**2*4,80)]
        ink=np.isin(labels,ids)
        # Prefer isolated strokes; a connected white halo is not a whole text line.
        score=sum(min(a,typical**2) for i,h,w,a in selected if i in ids)
        candidates.append((score,ink))
    ink=max(candidates,key=lambda item:item[0])[1]
    # A clipped effect/panel corner at the detection edge is not the last line
    # of a paragraph. Require a substantial blank gap before discarding it.
    rows=np.flatnonzero(ink.any(1))
    if len(rows)>10:
        gaps=np.flatnonzero(np.diff(rows)>max(8,crop.height*.045))
        if len(gaps):
            last=int(rows[gaps[-1]+1]);tail=ink[last:]
            xs=np.flatnonzero(tail.any(0))
            if rows[-1]>=crop.height-2 and len(xs) and xs[-1]-xs[0]<crop.width*.6 and last>crop.height*.7:ink[last:]=False
    return ink

def english_lines(crop,mask):
    ink=english_ink(crop,mask)
    return _ink_lines(ink)

def _ink_lines(ink):
    height,width=ink.shape
    projection=ink.sum(1)
    rows=projection>=max(2,float(projection.max())*.10);runs=[];start=None
    for y,active in enumerate([*rows,False]):
        if active and start is None:start=y
        elif not active and start is not None:
            if y-start>=3:runs.append([start,y])
            start=None
    # One- or two-pixel gutters are common in tightly set comic lettering.
    # They must not be closed: that used to join an entire paragraph into one row.
    _,_,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    heights=[h for x,y,w,h,a in stats[1:] if h>=7 and w>=3 and a>=12]
    typical=float(np.median(heights)) if heights else 20.
    merged=[]
    for a,b in runs:
        if b-a<typical*.35:continue
        if b-a>typical*1.65:
            # Descenders/italic overhang can bridge a gutter. Split at the
            # projection valley, bounded by the observed letter height.
            count=max(2,round((b-a)/typical))
            cuts=[a]
            for i in range(1,count):
                center=a+(b-a)*i/count;lo=max(cuts[-1]+int(typical*.5),int(center-typical*.3));hi=min(b-int(typical*.5),int(center+typical*.3))
                if hi>lo:cuts.append(lo+int(np.argmin(projection[lo:hi])))
            cuts.append(b);merged.extend(zip(cuts[:-1],cuts[1:]))
        else:merged.append((a,b))
    boxes=[]
    for a,b in merged:
        xs=np.flatnonzero(ink[a:b].any(0))
        if len(xs):boxes.append((max(0,int(xs[0])-2),max(0,a-2),min(width,int(xs[-1])+3),min(height,b+2)))
    return boxes

def english_strips(crop,mask,natural=False):
    """Deskew the paragraph before splitting tightly spaced italic lines.

    Recognition sees isolated original letter shapes on white; this removes
    screen tones and white outlines without depending on the balloon's shape.
    """
    original_ink=english_ink(crop,mask)
    ink=original_ink.astype(np.uint8)*255
    if not ink.any():return [],{},[]
    pad=max(12,round(max(crop.size)*.14));ink=np.pad(ink,pad)
    h,w=ink.shape;best=None
    for angle in range(-12,13):
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        rotated=cv2.warpAffine(ink,matrix,(w,h),flags=cv2.INTER_NEAREST)
        projection=(rotated>0).sum(1).astype(float)
        score=float(np.sum(projection**2))*(1-abs(angle)*.0005)
        if best is None or score>best[0]:best=(score,angle,rotated)
    _,angle,rotated=best;binary=rotated>0
    if natural:
        gray=np.asarray(crop.convert('L'))
        if np.median(gray[original_ink])>128:gray=255-gray
        gray=np.pad(gray,pad,constant_values=255)
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        gray=cv2.warpAffine(gray,matrix,(w,h),flags=cv2.INTER_CUBIC,borderValue=255)
        support=cv2.dilate(binary.astype(np.uint8),np.ones((3,3),np.uint8))>0
        natural_image=np.where(support,gray,255).astype(np.uint8)
    boxes=_ink_lines(binary)
    if len(boxes)>=3:
        typical=float(np.median([Y-y for x,y,X,Y in boxes]))
        boxes=[b for i,b in enumerate(boxes) if not (
            i in (0,len(boxes)-1) and b[2]-b[0]<typical*.75 and
            (b[0]<=pad+3 or b[2]>=w-pad-3))]
    strips=[];kept_boxes=[];ornaments=[]
    for x,y,X,Y in boxes:
        letter=binary[y:Y,x:X]
        if heart_shape(letter):
            matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
            inverse=cv2.invertAffineTransform(matrix)
            corners=np.array([[x,y,1],[X,y,1],[x,Y,1],[X,Y,1]])@inverse.T-pad
            ornaments.append([max(0,int(np.floor(corners[:,0].min()))),max(0,int(np.floor(corners[:,1].min()))),min(crop.width,int(np.ceil(corners[:,0].max()))),min(crop.height,int(np.ceil(corners[:,1].max())))])
            continue
        # A little padding avoids clipping italic overhang and punctuation.
        pixels=natural_image[y:Y,x:X] if natural else (~letter).astype(np.uint8)*255
        strips.append(Image.fromarray(np.pad(pixels,3,constant_values=255)).convert('RGB'))
        kept_boxes.append((x,y,X,Y))
    return strips,{'deskew_degrees':angle,'line_count':len(strips),'ornaments':ornaments},kept_boxes

def heart_shape(ink):
    """Two rounded top lobes and one tapered point; do not mistake V/U for hearts."""
    points=cv2.findNonZero(ink.astype(np.uint8))
    if points is None:return False
    x,y,w,h=cv2.boundingRect(points)
    if w<8 or h<8 or not .75<w/h<1.4:return False
    ink=ink[y:y+h,x:x+w].astype(np.uint8)
    contours,_=cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    body=np.zeros_like(ink);cv2.drawContours(body,contours,-1,1,cv2.FILLED)
    widths=body.sum(1);peak=max(widths)
    if widths[-max(2,h//6):].mean()>peak*.45 or widths[:max(1,h//12)].mean()>peak*.7:return False
    for row in body[1:max(3,h//3)]:
        if row[:w//3].any() and row[w*2//3:].any() and not row[w//2]:return True
    return False

def join_english(lines):
    # Join typographic hyphenation; ordinary line breaks become spaces, not requests.
    result=''
    for line in lines:
        line=line.strip()
        if not line:continue
        result=(result[:-1]+line) if result.endswith('-') and line[0].isalpha() else (result+' '+line).strip()
    return result
