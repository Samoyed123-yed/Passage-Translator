"""Independent full-page English detection/recognition, using local PP-OCRv6.

RapidOCR is Apache-2.0: https://github.com/RapidAI/RapidOCR
Model paths are explicit; this module never downloads or uploads anything.
"""
import re,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from paragraphs import english_ink,join_english
from typesetting import BubbleMap

class EnglishCoverage:
    def __init__(self,root):
        from rapidocr import RapidOCR
        model=Path(root)/'models/english-coverage'
        det=model/'PP-OCRv6_det_small.onnx';rec=model/'PP-OCRv6_rec_small.onnx'
        if not det.is_file() or not rec.is_file():raise FileNotFoundError('本地英文覆盖检测模型缺失。')
        self.reader=RapidOCR(params={'Global.use_cls':False,'Global.log_level':'error','Det.model_path':str(det),'Rec.model_path':str(rec),
            'Det.limit_side_len':1280,'Det.limit_type':'max','EngineConfig.onnxruntime.intra_op_num_threads':4,'EngineConfig.onnxruntime.inter_op_num_threads':1})

    def alternate_line(self,image,poly,text,score):
        """A second recognizer reads a rectified whole line, without row projection."""
        if score>=.92:return text,score
        import engine
        if not hasattr(self,'secondary'):self.secondary=engine.Vision().ocr('en')
        p=np.asarray(poly,np.float32)
        w=max(4,round(max(np.linalg.norm(p[0]-p[1]),np.linalg.norm(p[2]-p[3]))))
        h=max(4,round(max(np.linalg.norm(p[0]-p[3]),np.linalg.norm(p[1]-p[2]))))
        transform=cv2.getPerspectiveTransform(p,np.array([[0,0],[w,0],[w,h],[0,h]],np.float32))
        warped=cv2.warpPerspective(np.asarray(image.convert('RGB')),transform,(w,h),borderValue=(255,255,255))
        probs=self.secondary.session.run(None,{'x':engine._to_paddle_input(Image.fromarray(warped))})[0][0]
        prev=0;chars=[];scores=[]
        for row in probs:
            idx=int(row.argmax())
            if idx and idx!=prev and idx<len(self.secondary.charset):chars.append(self.secondary.charset[idx]);scores.append(float(row[idx]))
            prev=idx
        candidate=''.join(chars);confidence=float(np.mean(scores)) if scores else 0
        if confidence>.8 and confidence>score+.02 and re.search('[A-Za-z]{2}',candidate):return candidate,confidence
        return text,score

    def lines(self,image,minimum_score=.72,recover=False):
        result=self.reader(image)
        if result.boxes is None:return []
        lines=[]
        for poly,text,score in zip(result.boxes,result.txts,result.scores):
            if recover:text,score=self.alternate_line(image,poly,text,float(score))
            # Repeated O is often decoded as zero in an elongated interjection.
            text=re.sub(r'(?i)\b([a-z]+)0+\b',lambda m:m.group(0).replace('0','O'),text)
            letters=re.findall('[A-Za-z]',text)
            if score<minimum_score or len(letters)<2 or re.search('[\u4e00-\u9fff\u3040-\u30ff]',text):continue
            words=re.findall('[A-Za-z]+',text)
            if not words or (max(map(len,words))<2 and not re.fullmatch(r"[A-Za-z]['’][A-Za-z]+[.!?]*",text.strip())):continue
            poly=np.asarray(poly);x,y=poly.min(0);X,Y=poly.max(0)
            box=[max(0,int(x)-2),max(0,int(y)-2),min(image.width,int(X)+3),min(image.height,int(Y)+3)]
            glyph=min(np.linalg.norm(poly[1]-poly[0]),np.linalg.norm(poly[2]-poly[1]))
            lines.append({'box':box,'text':text,'score':float(score),'glyph':float(glyph),'poly':poly.tolist()})
        return lines

    def refine_group(self,image,group):
        """Retry the paragraph with context, so missed first/last lines can return."""
        if not group or min(l['score'] for l in group)>=.88:return group
        glyph=float(np.median([l['glyph'] for l in group]));bs=[l['box'] for l in group]
        x=min(b[0] for b in bs);y=min(b[1] for b in bs);X=max(b[2] for b in bs);Y=max(b[3] for b in bs)
        area=(max(0,int(x-glyph*.2)),max(0,int(y-glyph*1.1)),min(image.width,int(X+glyph*.2)),min(image.height,int(Y+glyph*.4)))
        crop=image.crop(area);candidates=self.lines(crop,minimum_score=.60,recover=True);refined=[]
        for line in candidates:
            if not glyph*.48<=line['glyph']<=glyph*2.1:continue
            a,b,c,d=line['box'];line['box']=[a+area[0],b+area[1],c+area[0],d+area[1]]
            line['poly']=(np.asarray(line['poly'])+[area[0],area[1]]).tolist()
            if min(X,line['box'][2])-max(x,line['box'][0])<min(X-x,line['box'][2]-line['box'][0])*.35:continue
            line['paragraph_retry']=True;refined.append(line)
        # Local detection may gain a missed line and lose a different one. Keep
        # unmatched originals rather than silently replacing the whole paragraph.
        for old in group:
            a,b,c,d=old['box'];matched=False
            for line in refined:
                u,v,U,V=line['box']
                intersection=max(0,min(c,U)-max(a,u))*max(0,min(d,V)-max(b,v))
                if intersection/max(1,min((c-a)*(d-b),(U-u)*(V-v)))>.42:matched=True;break
            if not matched:
                old=dict(old);old['text'],old['score']=self.alternate_line(image,old['poly'],old['text'],old['score']);refined.append(old)
        # Never replace a full paragraph with a shorter uncertain fragment.
        before=sum(len(l['text']) for l in group);after=sum(len(l['text']) for l in refined)
        if len(refined)>=len(group)-1 and before*.8<=after<=before*1.8 and (after>before or np.mean([l['score'] for l in refined])>np.mean([l['score'] for l in group])):return refined
        return group

    def detect(self,image):
        lines=self.lines(image);mask=Image.new('L',image.size);parents=list(range(len(lines)))
        shapes=[{'box':r['box']} for r in lines];bubbles=BubbleMap(image,shapes)
        labels=[(b or {}).get('label') for b in (bubbles.find(r) for r in shapes)]
        def root(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,a in enumerate(lines):
            x,y,X,Y=a['box'];cx=(x+X)/2;cy=(y+Y)/2
            for j in range(i):
                b=lines[j];u,v,U,V=b['box'];dy=abs(cy-(v+V)/2)
                if labels[i] is not None and labels[j] is not None and labels[i]!=labels[j]:continue
                glyph=max(a['glyph'],b['glyph']);small=min(a['glyph'],b['glyph'])
                overlap=max(0,min(X,U)-max(x,u))/max(1,min(X-x,U-u))
                gap=max(y,v)-min(Y,V)
                if overlap>=.48 and dy>small*.42 and gap<=glyph*.9 and glyph/max(1,small)<1.8:
                    parents[root(i)]=root(j)
        groups={}
        for i,line in enumerate(lines):groups.setdefault(root(i),[]).append(line)
        regions=[]
        for group in groups.values():
            group=self.refine_group(image,group)
            group.sort(key=lambda l:(l['box'][1],l['box'][0]));bs=[l['box'] for l in group]
            box=[min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)]
            source=join_english([l['text'] for l in group])
            # Treat short stand-alone sound effects as intentional skips.
            sound=bool(re.fullmatch(r'(?:(?:A+H+|H+A+|M+|N+G+H*|O+H+|U+W+O+A+H+|U+H+|H+M+)[!?.~…\s]*)+',source,re.I)) or bool(re.fullmatch(r'(?:SQUEEZE|SLURP|SPLASH|RUB|THRUST|PANT|SMACK|LICK|THROB|GULP|ZIP|HUMP)[!?.~…\s]*',source,re.I))
            r={'id':len(regions),'box':box,'source':source,'translation':'','source_language':'en','source_font_size':round(float(np.median([l['glyph'] for l in group]))/.8),
               'orientation':'horizontal','render_direction':'horizontal','direction_override':'horizontal','direction_uncertain':False,
               'kind':'sfx' if sound else 'dialogue','status':'跳过特效字' if sound else '待翻译','ocr_confidence':min(l['score'] for l in group),
               'ocr_lines':[l['text'] for l in group],'ocr_line_scores':[l['score'] for l in group],'detector':'PP-OCRv6 full-page','db_lines':group}
            regions.append(r)
            for line in group:
                b=line['box'];crop=image.crop(b);support=Image.new('L',crop.size,255)
                glyphs=english_ink(crop,support,expected_height=line['glyph'])
                # Retain the original grayscale stroke shape in a clean mask.
                current=np.asarray(mask.crop(b));mask.paste(Image.fromarray(np.maximum(current,glyphs.astype(np.uint8)*255)),b[:2])
        regions.sort(key=lambda r:(r['box'][1]//140,r['box'][0]))
        for i,r in enumerate(regions):r['id']=i
        return regions,mask

    def residuals(self,image,regions):
        lines=self.lines(image);hits=[]
        for r in regions:
            if r.get('status')!='已嵌字':continue
            x,y,X,Y=r['box'];words=set(re.findall('[A-Za-z]{3,}',r['source'].upper()))
            # Deliberately retained romanized names in the Chinese text are not failed erasure.
            names={s.upper() for s in re.findall(r'\b[A-Z][a-z]{2,}\b',r.get('translation',''))}
            words-=names
            for line in lines:
                a,b,c,d=line['box'];cx=(a+c)/2;cy=(b+d)/2
                if not (x-12<=cx<=X+12 and y-12<=cy<=Y+12):continue
                same=words & set(re.findall('[A-Za-z]{3,}',line['text'].upper()))
                if same and line['score']>=.8:hits.append({'id':r['id'],'words':sorted(same),'box':line['box']})
        return hits
