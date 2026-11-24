"""Native-resolution manga inpainting and measurable, bounded fallback cleaning."""
import math
import cv2
import numpy as np
from PIL import Image,ImageFilter

class MangaLama:
    def __init__(self,path,session_factory):self.session=session_factory(path)
    def inpaint(self,image,mask):
        # Tiles bound memory without shrinking original lettering or texture.
        w,h=image.size;output=image.copy();binary=np.asarray(mask)>0
        for y in range(0,h,768):
            for x in range(0,w,768):
                core=(x,y,min(w,x+768),min(h,y+768))
                if not binary[y:core[3],x:core[2]].any():continue
                a,b,c,d=max(0,x-96),max(0,y-96),min(w,core[2]+96),min(h,core[3]+96)
                rgb=np.asarray(image.crop((a,b,c,d)),np.float32)/255
                m=(np.asarray(mask.crop((a,b,c,d)))>0).astype(np.float32)
                H,W=math.ceil((d-b)/8)*8,math.ceil((c-a)/8)*8
                rgb=np.pad(rgb,((0,H-rgb.shape[0]),(0,W-rgb.shape[1]),(0,0)),mode='edge')
                m=np.pad(m,((0,H-m.shape[0]),(0,W-m.shape[1])))
                pred=self.session.run(None,{'image':rgb.transpose(2,0,1)[None],'mask':m[None,None]})[0][0].transpose(1,2,0)
                if pred.max()<=1.5:pred=pred*255
                tile=Image.fromarray(np.clip(pred,0,255).astype(np.uint8)).crop((x-a,y-b,core[2]-a,core[3]-b))
                output.paste(tile,(x,y),mask.crop(core))
        return output

def planar_fill(image,mask):
    rgb=np.asarray(image.convert('RGB'),np.float32);ink=np.asarray(mask)>0;h,w=ink.shape
    ring=cv2.dilate(ink.astype(np.uint8),np.ones((11,11),np.uint8))>0
    good=ring&~ink
    if good.sum()<30:return None
    yy,xx=np.mgrid[:h,:w];design=np.stack((xx/max(1,w),yy/max(1,h),np.ones_like(xx)),axis=-1)
    x=design[good];y=rgb[good]
    for _ in range(3):
        coeff=np.linalg.lstsq(x,y,rcond=None)[0];error=np.max(np.abs(x@coeff-y),axis=1)
        valid=error<max(8,float(np.percentile(error,75))*1.8)
        if valid.sum()<30:return None
        x,y=x[valid],y[valid]
    predicted=design@coeff
    errors=np.max(np.abs(predicted[good]-rgb[good]),axis=1)
    # Artwork and halftones must not be fitted as a flat speech balloon.
    if (errors<12).mean()<.96:return None
    return Image.fromarray(np.rint(np.clip(predicted,0,255)).astype(np.uint8))

def continuity(original,filled,mask):
    m=np.asarray(mask)>0;before=np.asarray(original.convert('L'),float);after=np.asarray(filled.convert('L'),float)
    outer=(cv2.dilate(m.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~m
    inner=m&~(cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    if not outer.any() or not inner.any():return False
    # Compare boundary gradients to the surrounding texture, not global mean color.
    grad=np.hypot(cv2.Sobel(after,cv2.CV_64F,1,0),cv2.Sobel(after,cv2.CV_64F,0,1))
    return bool(np.isfinite(after).all() and np.percentile(grad[inner],75)<=max(100,np.percentile(grad[outer],90)*1.8)
                and np.mean(np.abs(after[m]-before[m]))>2)

def clean_region(image,mask,r,fitted,bubble,keep,vision,regions=()):
    x,y,X,Y=r['box'];ref=r.get('source_font_size',24);pad=max(3,min(7,round(ref*.12)))
    area=(max(0,x-pad),max(0,y-pad),min(image.width,X+pad),min(image.height,Y+pad))
    a,b,c,d=area;source=image.crop(area);glyph=mask.crop(area).point(lambda v:255 if v>64 else 0)
    glyph=glyph.filter(ImageFilter.MaxFilter(pad*2+1))
    protected=np.asarray(keep.crop(area))>0
    for other in regions:
        if other is r:continue
        u,v,U,V=other['box'];u=max(a,u);v=max(b,v);U=min(c,U);V=min(d,V)
        if u<U and v<V:protected[v-b:V-b,u-a:U-a]=True
    if bubble:
        guard=Image.new('L',image.size);guard.paste(Image.fromarray(bubble['body']*255),bubble['box'][:2])
        protected|=np.asarray(guard.crop(area))==0
    m=(np.asarray(glyph)>0)&~protected
    if not m.any():return None,'missing-mask',{}
    glyph=Image.fromarray(m.astype(np.uint8)*255)
    fill=planar_fill(source,glyph);mode='local-planar-fill'
    if fill is not None:
        # A locally planar interior also exposes strokes missed by seg. Accept a
        # full-paragraph fit only when almost all non-ink pixels support the plane.
        rgb=np.asarray(source,dtype=float);prediction=np.asarray(fill,dtype=float)
        delta=np.max(np.abs(rgb-prediction),axis=2)
        domain=np.zeros_like(m);domain[y-b:Y-b,x-a:X-a]=True
        unmasked=domain&~m&~protected
        if unmasked.any() and (delta[unmasked]<12).mean()>.70:
            missing=domain&(delta>25)&~protected
            # Reject a large artwork silhouette, but retain thin complete strokes.
            _,labels,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
            # A burst / thought-balloon outline that the text box cuts through also differs
            # from the plane. Erasing it left a straight-edged white box over the outline,
            # so a long stroke running out of the text box is the balloon, not a glyph.
            dy0,dy1,dx0,dx1=y-b,Y-b,x-a,X-a
            def crosses(cx,cy,cw,ch):
                edge=cx<=dx0 or cy<=dy0 or cx+cw>=dx1 or cy+ch>=dy1
                return edge and max(cw,ch)>ref*1.2
            ids=[i for i,(cx,cy,cw,ch,area_) in enumerate(stats[1:],1)
                 if area_<domain.sum()*.15 and min(cw,ch)<max(16,ref*1.4) and not crosses(cx,cy,cw,ch)]
            m|=np.isin(labels,ids);glyph=Image.fromarray(m.astype(np.uint8)*255)
    if fill is None:
        context=(max(0,a-64),max(0,b-64),min(image.width,c+64),min(image.height,d+64))
        local=Image.new('L',(context[2]-context[0],context[3]-context[1]));local.paste(glyph,(a-context[0],b-context[1]))
        repaired=vision.inpaint(image.crop(context),local)
        fill=repaired.crop((a-context[0],b-context[1],c-context[0],d-context[1]));mode='manga-lama-native'
    clean=source.copy();clean.paste(fill,(0,0),glyph)
    residual=None
    if mode!='local-planar-fill' and hasattr(vision,'residual_ratio'):residual=vision.residual_ratio(clean,glyph)
    valid=(mode=='local-planar-fill' or continuity(source,clean,glyph)) and (residual is None or residual<.12)
    if not valid:
        # A compact opaque patch clears the source and supports the actual Chinese
        # ink. It is not the bounding rectangle of the balloon or context window.
        support=Image.new('L',source.size)
        support.paste(fitted['tile'],(fitted['position'][0]-a,fitted['position'][1]-b))
        support=support.filter(ImageFilter.MaxFilter(7))
        fallback=((np.asarray(support)>0)|m)&~protected
        glyph=Image.fromarray(fallback.astype(np.uint8)*255)
        clean=source.copy();clean.paste(Image.new('RGB',source.size,'white'),(0,0),glyph)
        mode='compact-white-fallback'
    return (clean,glyph,area),mode,{'residual_ratio_before_lettering':residual,'passed':valid,'fallback':not valid}
