"""Independent local bubble/text evidence; boxes seed contours, never erase areas."""
import cv2
import numpy as np

def coverage(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))/max(1,(a[2]-a[0])*(a[3]-a[1]))

class ComicEvidence:
    def __init__(self,model,session_factory):self.session=session_factory(model)
    def detect(self,image):
        # Overlapping native-aspect windows keep webtoon text visible at model scale.
        w,h=image.size;side=max(640,min(1600,round(min(w,h)*1.5)));step=max(1,round(side*.8));hits=[]
        xs=list(range(0,max(1,w-side+1),step));ys=list(range(0,max(1,h-side+1),step))
        xs=sorted(set(xs+[max(0,w-side)]));ys=sorted(set(ys+[max(0,h-side)]))
        for y in ys:
            for x in xs:
                crop=image.crop((x,y,min(w,x+side),min(h,y+side))).convert('RGB')
                pixels=np.asarray(crop.resize((640,640)),np.float32).transpose(2,0,1)[None]/255
                labels,boxes,scores=self.session.run(None,{'images':pixels,'orig_target_sizes':np.array([[crop.width,crop.height]],np.int64)})
                for label,box,score in zip(labels[0],boxes[0],scores[0]):
                    if score<.45 or int(label) not in (0,1,2):continue
                    a,b,c,d=box;box=[max(0,int(a+x)),max(0,int(b+y)),min(w,int(c+x)),min(h,int(d+y))]
                    if box[2]<=box[0] or box[3]<=box[1]:continue
                    hits.append({'class':int(label),'score':float(score),'box':box})
        result=[]
        for hit in sorted(hits,key=lambda a:-a['score']):
            if any(hit['class']==old['class'] and min(coverage(hit['box'],old['box']),coverage(old['box'],hit['box']))>.65 for old in result):continue
            result.append(hit)
        return result

def annotate(regions,hits,mask,line_mask=None):
    for r in regions:
        b=r['box'];matches=[(i,h) for i,h in enumerate(hits) if coverage(b,h['box'])>=.8]
        text=sorted(((i,h) for i,h in matches if h['class'] in (1,2)),key=lambda p:-p[1]['score'])
        bubble=sorted((h for _,h in matches if h['class']==0),key=lambda h:(h['box'][2]-h['box'][0])*(h['box'][3]-h['box'][1]))
        v={'class':text[0][1]['class'] if text else None,'score':text[0][1]['score'] if text else 0}
        if text:v.update(text_id=text[0][0],text_box=text[0][1]['box'])
        if bubble:v['bubble_box']=bubble[0]['box']
        if line_mask is not None:
            a,b,c,d=r['box'];ink=np.asarray(mask)[b:d,a:c]>96;line=np.asarray(line_mask)[b:d,a:c]>76
            v['line_support']=float((line&ink).sum()/max(1,ink.sum()))
        r['visual_evidence']=v
        if v['class']==1:r['boxed_prose']=True
    return regions

def recover_text_regions(regions,hits,mask,line_mask=None):
    """Recover a missed CTD block only when its independent text head has ink support."""
    for hit in hits:
        if hit['class'] not in (1,2) or hit['score']<.7:continue
        box=hit['box'];a,b,c,d=box
        if any(coverage(r['box'],box)>.5 or coverage(box,r['box'])>.5 for r in regions):continue
        ink=np.asarray(mask)[b:d,a:c]>96
        if ink.sum()<24 or ink.mean()<.015:continue
        if line_mask is not None:
            lines=np.asarray(line_mask)[b:d,a:c]>76
            if (ink&lines).sum()<ink.sum()*.2:continue
        regions.append({'box':box[:],'confidence':hit['score'],'detector':'RT-DETR text with CTD stroke support'})
    return regions

def contour(image,r,regions):
    seed=r.get('visual_evidence',{}).get('bubble_box')
    if not seed:return None
    a,b,c,d=seed;pad=12;a=max(0,a-pad);b=max(0,b-pad);c=min(image.width,c+pad);d=min(image.height,d+pad)
    gray=np.asarray(image.crop((a,b,c,d)).convert('L'))
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),45,120)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    x,y,X,Y=r['box'];candidates=[]
    for poly in contours:
        area=cv2.contourArea(poly)
        if area<(X-x)*(Y-y)*1.05 or area>gray.size*.96:continue
        px,py,pw,ph=cv2.boundingRect(poly)
        # Cropped/open edges are not evidence of a closed balloon.
        if px==0 or py==0 or px+pw>=gray.shape[1] or py+ph>=gray.shape[0]:continue
        body=np.zeros_like(gray);cv2.drawContours(body,[poly],-1,1,cv2.FILLED)
        ix,iy=max(x,a),max(y,b);iX,iY=min(X,c),min(Y,d)
        if iX<=ix or iY<=iy or body[iy-b:iY-b,ix-a:iX-a].sum()<.8*(X-x)*(Y-y):continue
        candidates.append((area,body))
    if not candidates:return None
    body=min(candidates,key=lambda p:p[0])[1]
    margin=max(2,round(min(c-a,d-b)*.02));safe=cv2.erode(body,np.ones((2*margin+1,2*margin+1),np.uint8))
    # Anchor to the source paragraph, not a bubble tail or the page center.
    cx,cy=(x+X)/2,(y+Y)/2;yy,xx=np.ogrid[b:d,a:c]
    for other in regions:
        if other is r:continue
        u,v,U,V=other['box'];ox,oy=(u+U)/2,(v+V)/2
        if a<=ox<c and b<=oy<d and body[int(oy-b),int(ox-a)]:
            safe[(xx-ox)**2+(yy-oy)**2<(xx-cx)**2+(yy-cy)**2+16]=0
    return {'box':[a,b,c,d],'safe':safe,'body':body,'center':(cx,cy),'label':tuple(seed),'bordered':True,'method':'rtdetr-local-contour'}

def effect_gate(r,page_font=0):
    """Require multiple negative visual clues. Lack of detection alone is not SFX."""
    import re
    v=r.get('visual_evidence',{})
    if r.get('boxed_voice') or r.get('boxed_prose') or v.get('class')==1:return False
    text=re.sub(r'[\s\W_]+','',r.get('source',''))
    if len(text)>8:return False
    votes=[]
    if v.get('line_support',1)<.30:votes.append('weak-line-support')
    if r.get('ocr_repetition_stopped'):votes.append('decoder-repetition')
    if r.get('ocr_confidence') is not None and r['ocr_confidence']<.5:votes.append('low-token-probability')
    if page_font and r.get('source_font_size',0)>page_font*1.9:votes.append('oversized-glyphs')
    if r.get('decorative_candidate'):votes.append('irregular-mask')
    if len(votes)<2:return False
    r.update(kind='sfx',translation='',status='跳过特效字',skip_review=True,visual_skip_reasons=votes)
    return True
