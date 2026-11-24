"""Local balloon contours and centered Chinese typesetting; no artwork generation."""
import math,re
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

_PAGE_MAPS=[]  # (image, white labels, white stats, dark) of the last few pages

def _page_map(image):
    """The white/black component maps depend only on the page pixels. Building them took
    ~0.9 s on a full page, and analysis asked once per text region; share them per image."""
    for entry in _PAGE_MAPS:
        if entry[0] is image:return entry
    rgb=np.asarray(image.convert('RGB'));hi=rgb.max(2);lo=rgb.min(2)
    white=((lo>218)&(hi-lo<28)).astype(np.uint8)
    # Never close the white mask: even a two-pixel closing can bridge a thin
    # black balloon outline and mistake adjacent white artwork for its interior.
    _,labels,stats,_=cv2.connectedComponentsWithStats(white,8)
    entry=[image,labels,stats,None]
    _PAGE_MAPS.insert(0,entry);del _PAGE_MAPS[2:]
    return entry

class BubbleMap:
    def __init__(self,image,regions):
        self.image=image; self.regions=regions; self.cache={}
        self._page=_page_map(image); self.labels,self.stats=self._page[1],self._page[2]
    @property
    def dark(self):return self._page[3]
    @dark.setter
    def dark(self,value):self._page[3]=value

    def find(self,r):
        from visual_evidence import contour
        seeded=contour(self.image,r,self.regions)
        if seeded is not None:return seeded
        if r.get('visual_evidence',{}).get('class')==2:return None
        found=self._search(r,self.labels,self.stats,False)
        if found is None:
            # White lettering in a solid black balloon (inner thoughts): the same search
            # over black areas, held to the stricter unbordered rules.
            if self.dark is None:
                rgb=np.asarray(self.image.convert('RGB'));hi=rgb.max(2)
                black=((hi<70)&(hi-rgb.min(2)<40)).astype(np.uint8)
                k=max(7,round(min(self.image.size)*.004))|1
                black=cv2.morphologyEx(black,cv2.MORPH_OPEN,np.ones((k,k),np.uint8))
                _,labels,stats,_=cv2.connectedComponentsWithStats(black,8);self.dark=(labels,stats)
            found=self._search(r,*self.dark,True)
        return found

    def _search(self,r,labels,stats,dark):
        x1,y1,x2,y2=map(int,r['box']); w=x2-x1; h=y2-y1
        ids,counts=np.unique(labels[y1:y2,x1:x2],return_counts=True)
        for i in np.argsort(-counts):
            label=int(ids[i]); x,y,bw,bh,area=map(int,stats[label])
            if not label or counts[i]<w*h*.22: continue
            # Open page margins and artwork backgrounds are not speech balloons.
            touches_edge=x<=1 or y<=1 or x+bw>=self.image.width-1 or y+bh>=self.image.height-1
            edge_count=sum((x<=1,y<=1,x+bw>=self.image.width-1,y+bh>=self.image.height-1))
            # Page gutters can be connected to white glyph outlines and artwork.
            # A dark boundary alone is not proof that this is a local balloon.
            if edge_count>=3 or (bw>.9*self.image.width and bh>.9*self.image.height):continue
            if area<w*h*.45 or area>self.image.width*self.image.height*.8:continue
            excessive_area=area>self.image.width*self.image.height*.15 or area>max(w*h*32,18000)
            if bw<w*.7 or bh<h*.7:continue
            # Joined white outlines around letters are not a white speech balloon.
            # A real balloon has breathing room outside the detected text block.
            margins=(x1-x,y1-y,x+bw-x2,y+bh-y2)
            if sum(v>max(4,min(w,h)*.06) for v in margins)<2:continue
            unusually_large=bw>max(w*9,350) or bh>max(h*5,400)
            if (dark,label) not in self.cache:
                raw=(labels[y:y+bh,x:x+bw]==label).astype(np.uint8)
                contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                filled=np.zeros_like(raw); cv2.drawContours(filled,contours,-1,1,cv2.FILLED)
                if not dark:filled=self._trim_screentone(raw,filled,(x1-x,y1-y,x2-x,y2-y))
                padded=np.pad(filled,1)
                ring=(cv2.dilate(padded,np.ones((3,3),np.uint8))>0)&(padded==0)
                boundary=np.asarray(self.image.crop((x-1,y-1,x+bw+1,y+bh+1)).convert('L'))
                yy,xx=np.ogrid[y-1:y+bh+1,x-1:x+bw+1]
                ring &= (yy>=0)&(yy<self.image.height)&(xx>=0)&(xx<self.image.width)
                bordered=False if dark else bool(ring.any() and (boundary[ring]<205).mean()>.6)
                margin=max(3,round(min(bw,bh)*.025)); k=margin*2+1
                safe=cv2.erode(filled,np.ones((k,k),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
                m=cv2.moments(safe)
                if not m['m00']: continue
                self.cache[(dark,label)]={'box':[x,y,x+bw,y+bh],'safe':safe,'body':filled,
                                   'center':(x+m['m10']/m['m00'],y+m['m01']/m['m00']),'label':label,'bordered':bordered}
            if (unusually_large or excessive_area) and not self.cache[(dark,label)]['bordered']:continue
            # A caption box whose outline is broken by a sparkle leaks into the panel's
            # white background, which runs out to the page edge; balloons do not.
            filled=float(self.cache[(dark,label)]['body'].sum())
            if touches_edge and filled>w*h*6:continue
            # A black thought balloon hugs its text; a larger black area is hair or clothing.
            if dark and filled>w*h*4.5:continue
            if touches_edge and not self.cache[(dark,label)]['bordered']:continue
            a,b=max(x,x1),max(y,y1);c,d=min(x+bw,x2),min(y+bh,y2)
            covered=self.cache[(dark,label)]['body'][b-y:d-y,a-x:c-x].sum() if c>a and d>b else 0
            if covered< w*h*.80:continue
            info=dict(self.cache[(dark,label)]); safe=info['safe'].copy()
            # Multiple utterances sharing a balloon retain separate local territories.
            peers=[]
            for other in self.regions:
                if other is r: continue
                a,b,c,d=other['box']; ox=int((a+c)/2); oy=int((b+d)/2)
                if x<=ox<x+bw and y<=oy<y+bh and info['body'][oy-y,ox-x]: peers.append((ox,oy))
            if peers:
                cx,cy=(x1+x2)/2,(y1+y2)/2
                yy,xx=np.ogrid[y:y+bh,x:x+bw]; own=(xx-cx)**2+(yy-cy)**2
                for ox,oy in peers: safe[(xx-ox)**2+(yy-oy)**2<own+25]=0
                info['center']=(cx,cy)
            info['safe']=safe
            return info
        return None

    def _trim_screentone(self,raw,filled,box):
        """The white between halftone dots joins the balloon through gaps in its outline;
        filling the outer contour then turns the whole screen into balloon interior.
        A small opening removes that lattice but keeps the solid white balloon."""
        k=max(5,round(min(self.image.size)*.003))|1
        opened=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((k,k),np.uint8))
        _,parts=cv2.connectedComponents(opened,connectivity=8)
        a,b,c,d=box;a,b=max(0,a),max(0,b)
        # Pieces with pixels at the text, not merely a bounding box around it: a white
        # ring left along a panel frame would otherwise refill the whole panel.
        near=np.unique(parts[max(0,b-k):d+k,max(0,a-k):c+k]);near=near[near>0]
        keep=np.isin(parts,near).astype(np.uint8)
        keep[b:d,a:c]|=filled[b:d,a:c]  # text may nearly touch the outline and cut the white
        contours,_=cv2.findContours(keep,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        trimmed=np.zeros_like(raw);cv2.drawContours(trimmed,contours,-1,1,cv2.FILLED)
        # Ordinary balloons barely change (anti-aliased rim, thin tail); keep them as they were.
        return trimmed if trimmed.sum()<filled.sum()*.85 else filled

@lru_cache(maxsize=256)
def get_font(path,size): return ImageFont.truetype(path,size)

def draw_cell(draw,x,y,char,font,symbol=None):
    """One vertical-text cell whose top edge is y. Every glyph keeps its place in the em
    box (一 sits mid-cell, 。、 in the upper right) instead of being pulled up to its own
    ink top; a full-height glyph lands exactly where an ink-top anchor would put it."""
    if symbol is not None and symbol is not font:
        draw.text((x,y),char,font=symbol,fill=255,anchor='mt');return
    draw.text((x,y-font.getbbox('国',anchor='ma')[1]),char,font=font,fill=255,anchor='ma')

def _chunks(text,count):
    # Balanced lines/columns; avoid placing closing punctuation at a new line start.
    result=[]; left=text
    for remaining in range(count,0,-1):
        n=math.ceil(len(left)/remaining)
        if n<len(left) and left[n-1].isascii() and left[n-1].isalpha() and left[n].isascii() and left[n].isalpha():
            start=n
            while start>0 and left[start-1].isascii() and left[start-1].isalpha():start-=1
            if start>0:n=start
            else:
                while n<len(left) and left[n].isascii() and left[n].isalpha():n+=1
        while n<len(left) and left[n] in '，。！？、；：）》」』】…!?.,': n+=1
        # An opening bracket belongs with its following text, never at a line end.
        while 1<n<len(left) and left[n-1] in '（《「『【“‘(':n-=1
        result.append(left[:n]); left=left[n:]
        if not left: break
    return result

def make_tile(text,size,count,vertical,font_path):
    font=get_font(str(font_path),size); lines=_chunks(text,count); sw=1
    symbol_path=Path(font_path).parent/'seguisym.ttf'
    symbol_font=get_font(str(symbol_path),size) if any(c in text for c in '♡♥❤') and symbol_path.is_file() else font
    if vertical:
        step=round(size*1.24); advance=round(size*1.13)
        tile=Image.new('L',(len(lines)*step+sw*4,max(map(len,lines))*advance+sw*4))
        draw=ImageDraw.Draw(tile)
        for col,line in enumerate(lines):
            for row,char in enumerate(line):
                # Vertical CJK punctuation uses the dedicated presentation forms.
                char=char.translate(str.maketrans('（）「」『』…—','︵︶﹁﹂﹃﹄︙︱'))
                draw_cell(draw,tile.width-sw*2-step/2-col*step,sw*2+row*advance,char,font,symbol_font if char in '♡♥❤' else None)
    else:
        # Use actual glyph height: em-size alone made Chinese paragraphs loose.
        # Keep a minimum gap and account for mixed Latin/symbol ascenders.
        glyph_height=max(font.getbbox(line)[3]-font.getbbox(line)[1] for line in lines)
        if symbol_font is not font:
            glyph_height=max(glyph_height,symbol_font.getbbox('♥')[3]-symbol_font.getbbox('♥')[1])
        step=max(round(size*1.08),glyph_height+max(2,round(size*.16)))
        tile=Image.new('L',(math.ceil(max(font.getlength(s) for s in lines))+sw*6,len(lines)*step+size//2+sw*4))
        draw=ImageDraw.Draw(tile)
        for row,line in enumerate(lines):
            if any(c in line for c in '♡♥❤'):
                start=(tile.width-font.getlength(line))/2
                for i,char in enumerate(line):
                    left=start+font.getlength(line[:i]);advance=font.getlength(char)
                    draw.text((left+advance/2,sw*2+row*step),char,font=symbol_font if char in '♡♥❤' else font,fill=255,anchor='mt')
            else:draw.text((tile.width/2,sw*2+row*step),line,font=font,fill=255,anchor='mt')
    box=tile.getbbox()
    return tile.crop(box) if box else None

def fit_text(text,source_box,vertical,font_path,bubble=None,source_text='',obstacles=(),enlarge=False,flow_aspect=None,reference_size=0):
    text=re.sub(r'\s+','',text)
    if not text: return None
    sx,sy,ex,ey=source_box; source_area=max(1,(ex-sx)*(ey-sy))
    if bubble:
        bx,by,bx2,by2=bubble['box']; safe=bubble['safe'].copy(); center=bubble['center']
    else:
        bx,by,bx2,by2=source_box
        safe=np.ones((by2-by,bx2-bx),np.uint8); safe[:2]=0; safe[-2:]=0; safe[:,:2]=0; safe[:,-2:]=0
        center=((sx+ex)/2,(sy+ey)/2)
    for a,b,c,d in obstacles:
        a=max(bx,a);b=max(by,b);c=min(bx2,c);d=min(by2,d)
        if c>a and d>b: safe[b-by:d-by,a-bx:c-bx]=0
    available=float(safe.sum()); w=bx2-bx; h=by2-by
    if available<20: return None
    # Preserve occupied area first; longer translations may use more of the balloon.
    expansion=min(1.65,max(1.,math.sqrt(len(text)/max(1,len(source_text)))))
    target=min(available*.72,source_area*.82*expansion)
    if enlarge and bubble and not reference_size and target/available<.38:
        # Small source boxes should not permanently cap Chinese at a tiny corner
        # of a verified balloon. Grow only sparse layouts, with a source-area cap.
        sparse_fraction=.40 if bubble.get('bordered') else .30
        target=min(available*.44,max(target,min(max(target*1.65,available*sparse_fraction),source_area*(5.0 if bubble.get('bordered') else 2.4))))
    maxsize=min(100,max(10,round(math.sqrt(max(target,25)/max(len(text),1))*1.6)))
    preserve_size=bool(reference_size and source_text and len(text)<=len(re.sub(r'\s+','',source_text)))
    if reference_size:maxsize=min(160,max(8,round(reference_size*1.05)))
    best=None; best_score=float('inf')
    source_aspect=flow_aspect or (ex-sx)/max(1,ey-sy)
    for size in range(maxsize,max(7,math.ceil(reference_size*.7))-1 if reference_size else 6,-1):
        maxcount=min(len(text),max(1,int((w if vertical else h)/(size*1.15))))
        for count in range(1,maxcount+1):
            tile=make_tile(text,size,count,vertical,font_path)
            if tile is None or tile.width>w-2 or tile.height>h-2: continue
            px=int(round(center[0]-tile.width/2)); py=int(round(center[1]-tile.height/2))
            lx,ly=px-bx,py-by
            if lx<0 or ly<0 or lx+tile.width>w or ly+tile.height>h: continue
            # Constrain the whole line block, including whitespace, to the true contour.
            allowed=safe[ly:ly+tile.height,lx:lx+tile.width]
            ink=np.asarray(tile)>0
            if not allowed[ink].all(): continue
            filled_area=tile.width*tile.height
            score=abs(math.log(max(1,filled_area)/max(1,target)))+.14*abs(math.log((tile.width/tile.height)/max(.1,source_aspect)))
            if size<10: score+=(10-size)*.12
            if reference_size and size<reference_size*.85:score+=(reference_size*.85/size-1)
            if reference_size:score+=8*abs(size/reference_size-1)
            if preserve_size and size<reference_size*.9:score+=10
            if score<best_score:
                best_score=score; best={'tile':tile,'position':(px,py),'size':size,'center':center,
                    'bounds':[px,py,px+tile.width,py+tile.height],'bubble':bubble is not None}
                best['vertical']=vertical; best['occupancy']=round(filled_area/available,4)
    return best

CLOSING='，。！？、；：）》」』】…—!?.,～~♡♥❤'
OPENING='（《「『【“‘('
VERTICAL_FORMS=str.maketrans('（）「」『』…—','︵︶﹁﹂﹃﹄︙︱')

def floating_space(image,mask,box):
    """Plain background around floating text (e.g. a stepped white caption box).

    Returns a boolean map over `box` that excludes artwork and frame lines, so
    vertical Chinese follows the same irregular shape the Japanese occupied.
    None means the background is not plain enough to trust; the whole box is used.
    """
    x,y,X,Y=map(int,box)
    gray=np.asarray(image.crop((x,y,X,Y)).convert('L')).astype(np.int16)
    glyph=cv2.dilate((np.asarray(mask.crop((x,y,X,Y)))>64).astype(np.uint8),np.ones((5,5),np.uint8))>0
    background=gray[~glyph]
    if background.size<50:return None
    similar=(np.abs(gray-np.median(background))<28)|glyph
    similar=cv2.morphologyEx(similar.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    if similar.mean()<.55:return None
    return cv2.erode(similar.astype(np.uint8),np.ones((3,3),np.uint8))>0

def _runs(flags):
    runs=[];start=None
    for i,v in enumerate([*flags,False]):
        if v and start is None:start=i
        elif not v and start is not None:runs.append((start,i));start=None
    return runs

def _column_lengths(text,caps,size,advance,lengths):
    """Balanced right-to-left fill. Closing punctuation never starts a column
    (it may hang half a cell), an opening bracket never ends one."""
    import linebreak
    n=len(text);greedy=0;pos=0;bounds=set();edge=0
    for t in linebreak.tokens(text):edge+=len(t);bounds.add(edge)
    for cap in caps:
        if pos>=n:break
        pos+=cap;greedy+=1
    if pos<n:return None
    result=[];pos=0
    for i in range(greedy):
        cap=caps[i];left=n-pos
        if left<=0:break
        take=min(cap,math.ceil(left/max(1,greedy-i)))
        while pos+take<n and text[pos+take] in CLOSING and (take+1)*advance-size*.45<=lengths[i]:take+=1
        while take>1 and pos+take<n and text[pos+take-1] in OPENING:take-=1
        while take>1 and pos+take<n and text[pos+take] in CLOSING:take-=1  # pull a character down instead
        if pos+take<n and pos+take not in bounds:
            # Never split a word or a name (天 / 咲光) when a boundary is one or two cells up.
            back=max((b for b in bounds if pos<b<pos+take),default=None)
            if back is not None and pos+take-back<=2:take=back-pos
        if take<=0:return None
        result.append(take);pos+=take
    return result if pos>=n else None

def fit_columns(text,box,font_path,bubble=None,obstacles=(),reference_size=0,source_text='',space=None):
    """Vertical Chinese placed where the Japanese was: right column first, each column
    starting at the top of the space at its own x and running down to the bottom.

    Columns follow an irregular contour by starting/ending where that x allows, but
    are never centered individually, so the reading order stays top-right first.
    The size stays near the source glyph size; shorter Chinese leaves space instead
    of shrinking, longer Chinese shrinks to at most 0.75x before giving up.
    """
    text=re.sub(r'\s+','',text)
    if not text:return None
    sx,sy,ex,ey=map(int,box)
    if bubble:
        bx,by,bX,bY=map(int,bubble['box']);allowed=bubble['safe'].astype(bool).copy()
    else:
        bx,by,bX,bY=sx,sy,ex,ey
        allowed=space.copy() if space is not None and space.shape==(ey-sy,ex-sx) else np.ones((ey-sy,ex-sx),bool)
    for a,b,c,d in obstacles:
        a=max(bx,int(a));b=max(by,int(b));c=min(bX,int(c));d=min(bY,int(d))
        if c>a and d>b and not (a<=sx and b<=sy and c>=ex and d>=ey):allowed[b-by:d-by,a-bx:c-bx]=False
    H,W=allowed.shape
    source_len=len(re.sub(r'[\W_]','',source_text)) or len(text)
    ref=reference_size or math.sqrt(max(1,(ex-sx)*(ey-sy))/max(1,source_len))*.85
    # Same size as the source glyphs: shorter Chinese leaves space, it is not enlarged.
    high=ref;low=ref if reference_size and len(re.sub(r'[\W♥♡❤_]','',text))<=source_len else ref*.75
    top_limit=sy-by;src_left=sx-bx;src_right=ex-bx;src_bottom=ey-by
    font_cache={}
    # In a balloon, first keep columns as long as the Japanese ones; only a
    # translation that does not fit may run on to the bottom of the balloon.
    passes=[(size,limit) for limit in ((True,False) if bubble else (False,))
            for size in range(max(8,round(high)),max(7,round(low))-1,-1)]
    for size,clip in passes:
        advance=round(size*1.08);pitch=round(size*1.32)
        def slots(right):
            found=[];x=right
            while x-size>=0 and len(found)<60:
                x0,x1=int(round(x-size)),int(round(x))
                best=None
                for a,b in _runs(allowed[:,max(0,x0-1):min(W,x1+1)].all(axis=1)):
                    start=max(a,top_limit-size//2) if bubble else a
                    if clip:b=min(b,src_bottom+advance//2)
                    if b-start>=advance and (best is None or start<best[0]):best=(start,b)
                if best:found.append((x0,x1,best[0],best[1]))
                elif found:break  # a hard edge ends the paragraph; never jump over artwork
                x-=pitch
                if not bubble and x-size<src_left-2:break
            # A sliver beside artwork (one or two cells far below the paragraph top)
            # would put the first words out of reading order. Drop such columns.
            if found:
                tallest=max(b-a for _,_,a,b in found);top=min(a for _,_,a,_ in found)
                found=[s for s in found if s[3]-s[2]>=max(2*advance,tallest*.45) and s[2]-top<=max(3*advance,tallest*.35)]
            return found
        for anchor in ('center','right'):
            right=src_right if anchor=='right' else None
            if right is None:
                probe=slots(src_right)
                caps=[max(0,(b-a)//advance) for _,_,a,b in probe]
                lengths=_column_lengths(text,caps,size,advance,[b-a for _,_,a,b in probe]) if probe else None
                if not lengths:continue
                block=len(lengths)*pitch-(pitch-size)
                right=min(W,int(round((src_left+src_right)/2+block/2)))
                if right<=src_right:continue  # centering would not move it; use right anchor
            found=slots(right)
            caps=[max(0,(b-a)//advance) for _,_,a,b in found]
            lengths=_column_lengths(text,caps,size,advance,[b-a for _,_,a,b in found]) if found else None
            if not lengths:continue
            key=(str(font_path),size)
            if key not in font_cache:font_cache[key]=get_font(str(font_path),size)
            font=font_cache[key]
            symbol_path=Path(font_path).parent/'seguisym.ttf'
            symbol=get_font(str(symbol_path),size) if symbol_path.is_file() and any(c in text for c in '♡♥❤') else font
            tile=Image.new('L',(W,H));draw=ImageDraw.Draw(tile);pos=0
            for (x0,x1,start,_),count in zip(found,lengths):
                for j,char in enumerate(text[pos:pos+count]):
                    glyph=char.translate(VERTICAL_FORMS)
                    draw_cell(draw,(x0+x1)/2,start+j*advance,glyph,font,symbol if char in '♡♥❤' else None)
                pos+=count
            ink=np.asarray(tile)>0
            if not ink.any() or (ink&~allowed).sum()>ink.sum()*.02:continue
            l,t,r,b=tile.getbbox()
            return {'tile':tile.crop((l,t,r,b)),'position':(bx+l,by+t),'size':size,'center':((bx*2+l+r)/2,(by*2+t+b)/2),
                    'bounds':[bx+l,by+t,bx+r,by+b],'bubble':bubble is not None,'vertical':True,
                    'occupancy':round((r-l)*(b-t)/max(1,float(allowed.sum())),4),'anchor':'source-columns-'+anchor,
                    'columns':lengths}
    return None

# Proportions measured on a scanlation group's edition (272 balloons, 2026-11-20):
# column pitch 1.16 em, character advance 1.0 em, Chinese 1.15x the Japanese glyph,
# block centred on the balloon centroid (offset < 1%), >= 0.24 em clear of the edge.
PITCH, ADVANCE, SOURCE_SCALE, MIN_SCALE, MARGIN = 1.19, 1.04, 1.2, .72, .12
VERTICAL_PUNCT = str.maketrans('（）「」『』…—，。、：；', '︵︶﹁﹂﹃﹄︙︱︐︒︑︓︔')

def _lines_for(text, n, emphasis):
    """n lines of (string, scale). Emphasised span gets its own line(s) at 1.3x."""
    import linebreak
    if emphasis:
        a, b = emphasis; pre, emph, post = text[:a], text[a:b], text[b:]
        if post and not re.sub(r'[\W♥♡❤]', '', post): emph, post = emph + post, ''
        if pre and not re.sub(r'[\W♥♡❤]', '', pre): emph, pre = pre + emph, ''
        out = []
        for part, scale, share in ((pre, 1., None), (emph, 1.3, None), (post, 1., None)):
            if not part: continue
            k = max(1, min(len(linebreak.tokens(part)), round(n * len(part) / max(1, len(text)))))
            lines = linebreak.partition(part, k) or linebreak.char_partition(part, k)
            out.extend((s, scale) for s in lines)
        return out
    lines = linebreak.partition(text, n)
    if lines is None:
        if n > len(text): return None
        lines = linebreak.char_partition(text, n)
    return [(s, 1.) for s in lines]

def fit_balloon(text, box, font_path, bubble=None, obstacles=(), reference_size=0, source_text='', space=None,
                emphasis=None, emph_font=None, horizontal=False, fixed_size=None, relaxed=False):
    """Centred, word-broken lettering (vertical, or horizontal with horizontal=True) the way a scanlation typesetter sets it.

    The block (right-to-left columns, each top-aligned) is centred on the balloon's
    centroid, or on the source paragraph when there is no balloon; size starts at
    1.15x the Japanese glyph and only shrinks (to 0.72x) when the block cannot keep a
    0.22 em margin. Among line counts that fit, the one whose proportions match the
    balloon interior wins, like a human balancing the shape.
    """
    import linebreak
    text = re.sub(r'\s+', '', text)
    if not text: return None
    sx, sy, ex, ey = map(int, box)
    if bubble:
        bx, by, bX, bY = map(int, bubble['box']); allowed = bubble['safe'].astype(bool).copy()
    else:
        bx, by, bX, bY = sx - 4, sy - 4, ex + 4, ey + 4
        allowed = np.ones((bY - by, bX - bx), bool)
        if space is not None and space.shape == (ey - sy, ex - sx):
            allowed[:] = False; allowed[4:4 + space.shape[0], 4:4 + space.shape[1]] = space
    for a, b, c, d in obstacles:
        a = max(bx, int(a)); b = max(by, int(b)); c = min(bX, int(c)); d = min(bY, int(d))
        if c > a and d > b and not (a <= sx and b <= sy and c >= ex and d >= ey): allowed[b - by:d - by, a - bx:c - bx] = False
    H, W = allowed.shape
    if allowed.sum() < 50: return None
    m = cv2.moments(allowed.astype(np.uint8))
    ax, ay = (m['m10'] / m['m00'], m['m01'] / m['m00']) if bubble else ((sx + ex) / 2 - bx, (sy + ey) / 2 - by)
    shown = allowed
    if horizontal:
        # Lay out in a rotated frame: layout column u = H-1-y (first line on top), layout row
        # v = x (left to right). Lines are then the vertical columns of the same algorithm.
        allowed = np.ascontiguousarray(allowed[::-1, :].T); ax, ay = H - 1 - ay, ax; H, W = allowed.shape
    ys, xs = np.nonzero(allowed); avail_w, avail_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    target_aspect = (avail_w * .60) / max(1, avail_h * .73)
    source_len = len(re.sub(r'[\W_]', '', source_text)) or len(text)
    ref = reference_size or math.sqrt(max(1, (ex - sx) * (ey - sy)) / max(1, source_len)) * .85
    # Short cries/replies in small balloons are set larger, as the group does (up to 1.4x).
    short = len(re.sub(r'[\W♥♡❤]', '', text)) <= 6
    # Larger-than-source type is a measured habit of Japanese scanlations only; Korean and
    # English sources never get Chinese bigger than the original lettering (user rule).
    japanese = not source_text or re.search('[぀-ヿ一-鿿]', source_text) and not re.search('[A-Za-z]', source_text)
    hi = max(8, round(ref * ((1.4 if short and bubble else SOURCE_SCALE) if japanese else 1.))); lo = max(7, round(ref * MIN_SCALE))
    # User rule: Chinese no longer than the Japanese keeps at least the Japanese size.
    # When the normal margin cannot hold it, the margin and spacing tighten instead.
    no_shrink = bool(reference_size) and len(re.sub(r'[\W♥♡❤_]', '', text)) <= source_len
    if no_shrink: lo = max(lo, round(reference_size))
    if fixed_size: hi = lo = int(fixed_size)
    margin_scale = (.04 if relaxed else MARGIN) if bubble else (.0 if relaxed else .06)
    pitch_k, advance_k = (1.08, 1.0) if relaxed else (PITCH, ADVANCE)
    first_fit = hi
    fill_len = .55 if horizontal else .74  # horizontal lines run ~55% of the balloon width
    def extra(lines, size):
        # Smaller-than-largest size costs a little; extra columns and a stranded short
        # column ("然后" alone between long ones) cost more.
        lens = [len(re.sub(r'[\W♥♡❤]', '', s)) for s, _ in lines]
        stranded = len(lens) > 2 and min(lens) <= 2 and max(lens) >= 4
        lone = sum(1 for n in lens if n == 1) if len(lens) > 1 else 0  # one-character lines (我 / 先) read badly
        return 1.5 * math.log(hi / size) + .1 * max(0, len(lines) - 2) + (.5 if stranded else 0) + .6 * lone
    max_lines = min(8, len(linebreak.tokens(text)) + (1 if emphasis else 0))
    options = []
    for size in range(hi, lo - 1, -1):
        # Look a little below the largest fitting size: the group prefers fewer, longer
        # columns at a slightly smaller size over many short ones.
        if options and size < first_fit * .86: break
        pad = max(1, round(size * margin_scale))
        fit = cv2.erode(allowed.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) if pad else allowed.astype(np.uint8)
        integ = cv2.integral(fit)
        def inside(x0, y0, x1, y1):
            x0, y0, x1, y1 = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H: return False
            return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0] == (x1 - x0) * (y1 - y0)
        fit_bool = fit > 0
        def runs_of(flags):
            out, start = [], None
            for i, v in enumerate([*flags, False]):
                if v and start is None: start = i
                elif not v and start is not None: out.append((start, i)); start = None
            return out
        def pick_run(runs, at):
            hit = [r for r in runs if r[0] <= at < r[1]]
            return hit[0] if hit else (max(runs, key=lambda r: r[1] - r[0]) if runs else None)
        def free_place(lines, sizes):
            """Scanlation-style free placement: columns spread evenly across the width at the
            text's centre row (equal side margins), each column centred in the vertical space
            open at its own x, with a gentle right-high / left-low stagger. None if it can't."""
            n = len(lines); heights = [len(s) * sz * advance_k for (s, _), sz in zip(lines, sizes)]
            row = pick_run(runs_of(fit_bool[min(H - 1, max(0, int(round(ay))))]), int(round(ax)))
            if row is None: return None
            L, R = row; ink_w = sum(sizes); span = R - L
            gap = min(max((span - ink_w) / (n + 1), (pitch_k - 1) * min(sizes)), (.9 if n <= 2 else .5) * max(sizes))
            total = ink_w + gap * (n - 1)
            if total > span: return None
            right = min(R, max(L + total, ax + total / 2))
            cols, x = [], right
            for i, ((s, _), sz, h) in enumerate(zip(lines, sizes, heights)):
                x0, x1 = x - sz, x
                a, b = int(math.floor(x0)), int(math.ceil(x1))
                if a < 0 or b > W: return None
                run = pick_run(runs_of(fit_bool[:, a:b].all(axis=1)), int(round(ay)))
                if run is None or run[1] - run[0] < h: return None
                slack = (run[1] - run[0] - h) / 2
                shift = (i - (n - 1) / 2) * min(.55 * sz, slack / max(1, (n - 1) / 2)) if n > 1 and not horizontal else 0
                top = min(max((run[0] + run[1]) / 2 - h / 2 + shift, run[0]), run[1] - h)
                cols.append((x0, top, x1, top + h)); x -= sz + gap
            return cols
        before = len(options)
        for n in range(1, max_lines + 1):
            lines = _lines_for(text, n, emphasis)
            if not lines: continue
            sizes = [size * s for _, s in lines]
            widths = [sz * pitch_k for sz in sizes]; block_w = sum(widths) - (widths[-1] - sizes[-1])
            block_h = max(len(s) * sz * advance_k for (s, _), sz in zip(lines, sizes))
            # Free spread only for one or two short columns (split lobes); longer text stays a compact block.
            cols = free_place(lines, sizes) if bubble and n <= 2 else None
            if cols and all(inside(*c) for c in cols):
                bw = max(c[2] for c in cols) - min(c[0] for c in cols); bh = max(c[3] for c in cols) - min(c[1] for c in cols)
                score = (abs(math.log(block_h / max(1, avail_h * fill_len))) + .6 * abs(math.log(bw / max(1, avail_w * .60)))
                         + .04 * n - .12 * linebreak.quality([s for s, _ in lines]))
                options.append((score + extra(lines, size), size, lines, sizes, cols)); continue
            if block_w > avail_w or block_h > avail_h: continue
            for dx, dy in ((0, 0), (0, -.06), (0, .06), (-.06, 0), (.06, 0), (0, -.12), (0, .12), (-.12, 0), (.12, 0)):
                right = ax + block_w / 2 + dx * avail_w; top = ay - block_h / 2 + dy * avail_h
                cols, x = [], right
                # Compact block whose columns step down slightly to the left, as the group sets it.
                step = .3 * size if n > 2 and not horizontal else 0; top -= step * (n - 1) / 2
                for i, ((s, _), sz, wdt) in enumerate(zip(lines, sizes, widths)):
                    h = len(s) * sz * advance_k; t = top + i * step + ((block_h - h) / 2 if horizontal else 0)  # horizontal lines are centred
                    cols.append((x - sz, t, x, t + h)); x -= wdt
                if all(inside(*c) for c in cols):
                    # The group fills ~74% of the balloon height and ~60% of its width.
                    score = (abs(math.log(block_h / max(1, avail_h * fill_len))) + .6 * abs(math.log(block_w / max(1, avail_w * .60)))
                             + 2.5 * math.hypot(dx, dy) + .04 * n - .12 * linebreak.quality([s for s, _ in lines]))
                    options.append((score + extra(lines, size), size, lines, sizes, cols)); break
        if len(options) > before and not before: first_fit = size
    for _, size, lines, sizes, cols in sorted(options, key=lambda o: o[0]):
        tile = Image.new('L', (shown.shape[1], shown.shape[0])); draw = ImageDraw.Draw(tile)
        for (s, scale), sz, (x0, y0, x1, y1) in zip(lines, sizes, cols):
            path = emph_font if (scale > 1 and emph_font) else font_path
            font = get_font(str(path), max(6, round(sz)))
            symbol = font
            if font.getbbox('♥') == font.getbbox('￿'):  # no heart in this font: Segoe UI Symbol, enlarged
                symbol_path = Path('C:/Windows/Fonts/seguisym.ttf')
                if symbol_path.is_file(): symbol = get_font(str(symbol_path), max(6, round(sz * 1.25)))
            for j, ch in enumerate(s):
                face = symbol if ch in '♡♥❤' else font
                if horizontal:  # back from the rotated frame: line centre y = H-1-centre u
                    draw.text((y0 + j * sz * advance_k + sz / 2, shown.shape[0] - 1 - (x0 + x1) / 2), ch, font=face, fill=255, anchor='mm')
                else:
                    draw_cell(draw, (x0 + x1) / 2, y0 + j * sz * advance_k, ch.translate(VERTICAL_PUNCT), font, face if face is not font else None)
        ink = np.asarray(tile) > 0
        if not ink.any() or (ink & ~shown).sum() > ink.sum() * .03: continue
        l, t, r, b = tile.getbbox()
        cx, cy = (ay, H - 1 - ax) if horizontal else (ax, ay)
        return {'tile': tile.crop((l, t, r, b)), 'position': (bx + l, by + t), 'size': size, 'center': (bx + cx, by + cy),
                'bounds': [bx + l, by + t, bx + r, by + b], 'bubble': bubble is not None, 'vertical': not horizontal,
                'occupancy': round((r - l) * (b - t) / max(1, float(allowed.sum())), 4), 'anchor': 'balloon-centred',
                'columns': [len(s) for s, _ in lines], 'lines': [s for s, _ in lines]}
    if no_shrink and not relaxed and not fixed_size:
        # Floating text over artwork: the plain-background map can be too ragged for any
        # column, yet the Japanese itself filled this box. Use the box before tightening.
        if bubble is None and space is not None:
            return fit_balloon(text, box, font_path, None, obstacles, reference_size, source_text, None, emphasis, emph_font, horizontal)
        return fit_balloon(text, box, font_path, bubble, obstacles, reference_size, source_text, space, emphasis, emph_font, horizontal, None, True)
    return None

def fit_lobes(text, lobes, font_path, bubble, source_text, obstacles=(), reference_size=0):
    """A joined balloon of two shapes (high right lobe, low left lobe) carries two groups of
    Japanese columns. Set the Chinese the same way: the first part in the right lobe, the
    rest in the left one, each at least the Japanese size, instead of one centred block
    squeezed into the neck between them. None if either part does not fit."""
    import linebreak
    text = re.sub(r'\s+', '', text); src = re.sub(r'[\W_]', '', source_text)
    if len(lobes) != 2 or not text or not src: return None
    right, left = sorted(lobes, key=lambda b: -b[0])
    if right[1] >= left[1]: return None  # only the usual right-high / left-low order
    area = [(b[2] - b[0]) * (b[3] - b[1]) for b in (right, left)]
    ratio = area[0] / max(1, sum(area))
    toks = linebreak.tokens(text)
    if len(toks) < 2: return None
    best, cut, run = None, None, 0
    for j in range(1, len(toks)):
        run += len(toks[j - 1])
        cost = 6 * abs(run / len(text) - ratio) - .5 * linebreak._break_bonus(toks[j - 1], toks[j])
        if best is None or cost < best: best, cut = cost, j
    parts = [''.join(toks[:cut]), ''.join(toks[cut:])]
    n0 = max(1, round(len(src) * ratio)); sources = [src[:n0], src[n0:] or src[-1:]]
    bx, by, bX, bY = map(int, bubble['box']); safe = bubble['safe'].astype(bool)
    mid = round((right[0] + left[2]) / 2); ref = reference_size or 24
    halves = [(max(bx, mid), bX), (bx, min(bX, mid))]
    fits = []
    for part, source, lobe, (lo_x, hi_x) in zip(parts, sources, (right, left), halves):
        sx = max(lo_x, int(lobe[0] - .8 * ref)); ex = min(hi_x, int(lobe[2] + .8 * ref))
        sy = max(by, int(lobe[1] - 1.2 * ref)); ey = min(bY, int(lobe[3] + 1.2 * ref))
        if ex - sx < ref or ey - sy < ref: return None
        space = safe[sy - by:ey - by, sx - bx:ex - bx]
        fitted = fit_balloon(part, (sx, sy, ex, ey), font_path, None, obstacles, reference_size, source, space)
        if not fitted or not fitted.get('vertical', True): return None
        fits.append(fitted)
    x = min(f['bounds'][0] for f in fits); y = min(f['bounds'][1] for f in fits)
    X = max(f['bounds'][2] for f in fits); Y = max(f['bounds'][3] for f in fits)
    if fits[0]['bounds'][1] > fits[1]['bounds'][1]: return None  # keep the reading order visible
    ink = np.zeros((Y - y, X - x), np.uint8)
    for f in fits:
        a, b = f['position'][0] - x, f['position'][1] - y; t = np.asarray(f['tile'])
        ink[b:b + t.shape[0], a:a + t.shape[1]] = np.maximum(ink[b:b + t.shape[0], a:a + t.shape[1]], t)
    return {'tile': Image.fromarray(ink), 'position': (x, y), 'bounds': [x, y, X, Y], 'size': min(f['size'] for f in fits),
            'center': bubble['center'], 'bubble': True, 'vertical': True, 'anchor': 'two-lobes',
            'occupancy': round((X - x) * (Y - y) / max(1, float(safe.sum())), 4),
            'columns': fits[0].get('columns', []) + fits[1].get('columns', []), 'lines': fits[0].get('lines', []) + fits[1].get('lines', [])}

def fit_adaptive(text,box,source_vertical,font_path,bubble=None,source_text='',obstacles=(),override='auto',reference_size=0,space=None,emphasis=None,emph_font=None):
    bounds=box
    ratio=(bounds[2]-bounds[0])/max(1,bounds[3]-bounds[1])
    vertical=source_vertical
    if override=='auto' and len(re.sub(r'\s+','',text))<=24 and ratio>=1.6 and (not source_vertical or (box[2]-box[0])/max(1,box[3]-box[1])>=.85 or ratio>=2.4):
        vertical=False
    if override in ('horizontal','vertical'):vertical=override=='vertical'
    if vertical:
        centred=fit_balloon(text,box,font_path,bubble,obstacles,reference_size,source_text,space,emphasis,emph_font)
        if centred:return centred
        columns=fit_columns(text,box,font_path,bubble,obstacles,reference_size,source_text,space)
        if columns:return columns
    preferred_aspect=ratio if vertical!=source_vertical else (box[2]-box[0])/max(1,box[3]-box[1])
    fitted=fit_text(text,box,vertical,font_path,bubble,source_text,obstacles,True,preferred_aspect,reference_size)
    if override=='auto' and vertical!=source_vertical:
        original=fit_text(text,box,source_vertical,font_path,bubble,source_text,obstacles,True,None,reference_size)
        if original and (not fitted or (ratio<2.4 and fitted['size']<original['size']*1.15)):
            fitted=original;vertical=source_vertical
    if bubble and reference_size and len(re.sub(r'\s+','',text))<=len(re.sub(r'\s+','',source_text)) and (not fitted or fitted['size']<reference_size):
        # A tail or indentation can put the balloon centroid in a narrow neck.
        # Try the original text center using the SAME contour and obstacles.
        anchored=dict(bubble,center=((box[0]+box[2])/2,(box[1]+box[3])/2))
        alternative=fit_text(text,box,vertical,font_path,anchored,source_text,obstacles,True,preferred_aspect,reference_size)
        if alternative and (not fitted or alternative['size']>fitted['size']):
            fitted=alternative
            fitted['anchor']='source-center'
    if bubble and float(bubble['safe'].mean())<.85:
        flowing=fit_contour(text,box,vertical,font_path,bubble,obstacles,reference_size or (fitted['size'] if fitted else 24))
        if flowing and (not fitted or flowing['size']>fitted['size']*1.12):fitted=flowing
    if fitted and bubble and not fitted.get('vertical') and fitted.get('anchor')!='source-center':
        # Horizontal balloons: the established rules above pick the size; the lines are then
        # set the scanlation way (word breaks, centred lines, tight spacing) at that size.
        lettered=fit_balloon(text,box,font_path,bubble,obstacles,reference_size,source_text,space,emphasis,emph_font,horizontal=True,fixed_size=fitted['size'])
        if lettered:return lettered
    return fitted

def fit_contour(text,box,vertical,font_path,bubble,obstacles,reference):
    """Stagger independent columns/rows within the unchanged balloon contour."""
    text=re.sub(r'\s+','',text)
    if not text:return None
    bx,by,bX,bY=bubble['box'];safe=bubble['safe'].copy();h,w=safe.shape
    for a,b,c,d in obstacles:
        a=max(a,bx);b=max(b,by);c=min(c,bX);d=min(d,bY)
        if c>a and d>b:safe[b-by:d-by,a-bx:c-bx]=0
    cx,cy=bubble['center'];cx-=bx;cy-=by
    for size in range(min(100,max(10,round(reference))),max(7,min(60,round(reference*.7)))-1,-1):
        step=round(size*1.24)
        for count in range(1,min(len(text),8)+1):
            lines=_chunks(text,count);placed=[]
            for i,line in enumerate(lines):
                tile=make_tile(line,size,1,vertical,font_path)
                if tile is None:break
                tw,th=tile.size
                if tw>w or th>h:break
                main=(cx+(len(lines)-1-2*i)*step/2-tw/2) if vertical else (cy+(2*i-len(lines)+1)*step/2-th/2)
                main=round(main)
                if main<0 or main+(tw if vertical else th)>(w if vertical else h):break
                limit=(h-th) if vertical else (w-tw)
                ideal=cy-th/2 if vertical else cx-tw/2
                candidates=sorted(range(0,limit+1,2),key=lambda v:abs(v-ideal))
                found=None;ink=np.asarray(tile)>0
                for shift in candidates:
                    x,y=(main,shift) if vertical else (shift,main)
                    if safe[y:y+th,x:x+tw][ink].all():found=(x,y,tile);break
                if found is None:break
                placed.append(found)
            if len(placed)!=len(lines):continue
            x=min(p[0] for p in placed);y=min(p[1] for p in placed)
            X=max(a+t.width for a,b,t in placed);Y=max(b+t.height for a,b,t in placed)
            tile=Image.new('L',(X-x,Y-y))
            for a,b,t in placed:tile.paste(t,(a-x,b-y))
            return {'tile':tile,'position':(bx+x,by+y),'bounds':[bx+x,by+y,bx+X,by+Y],
                    'size':size,'center':bubble['center'],'bubble':True,'vertical':vertical,
                    'occupancy':round((X-x)*(Y-y)/max(1,float(safe.sum())),4),'anchor':'contour-flow'}
    return None
