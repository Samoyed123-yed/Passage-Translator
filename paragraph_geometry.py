"""Associate crossing glyphs with a single detected paragraph before OCR/render.

No prose/effect classification is inferred here. Disconnected missed paragraphs
and ambiguous shared components remain the responsibility of later detection.
"""
import cv2
import numpy as np

def source_ink_size(region, mask):
    """DB line maps are eroded probability cores, not full character widths.

    Measure the full segmentation ink along the already inferred reading lines.
    Do not let a merged multi-column block become a giant font estimate.
    """
    vertical=region.get('orientation')=='vertical'
    if region.get('detector')=='large glyph-mask line' and not vertical:
        # Found as one row of large glyphs (see engine.large_mask_lines); the row height
        # is the glyph size, while DB cores see only the faint gradient fill.
        return round((region['box'][3]-region['box'][1]-8)*.85),'large-glyph-row'
    cores=region.get('paragraph_geometry',{}).get('det_lines',[])
    cores=[b for b in cores if (b[3]-b[1] if vertical else b[2]-b[0])>1.5*(b[2]-b[0] if vertical else b[3]-b[1])]
    if cores:
        # Close-set columns may merge in projection, but the independent DB cores
        # retain their individual centers. Measure full ink in each center's lane.
        axis=0 if vertical else 1
        cores.sort(key=lambda b:(b[axis]+b[axis+2])/2)
        centers=[(b[axis]+b[axis+2])/2 for b in cores]
        box=region['box'];ink=np.asarray(mask)>96;measured=[]
        for i,b in enumerate(cores):
            center=centers[i];width=b[axis+2]-b[axis]
            if i and center-centers[i-1]<max(4,width*.6):continue
            low=box[axis] if i==0 else (centers[i-1]+center)/2
            high=box[axis+2] if i==len(cores)-1 else (center+centers[i+1])/2
            a,c=max(box[axis],int(round(low))),min(box[axis+2],int(round(high)))
            cross=1-axis;pad=max(4,width)
            start=max(box[cross],b[cross]-pad);end=min(box[cross+2],b[cross+2]+pad)
            local=ink[start:end,a:c] if vertical else ink[a:c,start:end]
            projection=local.any(axis=0 if vertical else 1) if local.size else []
            positions=np.flatnonzero(projection)
            if len(positions):
                actual=int(positions[-1]-positions[0]+1)
                # Furigana beside a column is a separate narrow run; it is not glyph width.
                cuts=np.flatnonzero(np.diff(positions)>2)
                if len(cuts):
                    runs=sorted((int(r[-1]-r[0]+1) for r in np.split(positions,cuts+1)),reverse=True)
                    if runs[0]>=actual*.55 and runs[1]<runs[0]*.6:actual=runs[0]
                if 8<=actual<=140:measured.append(actual)
        if measured:return round(float(np.median(measured))),'full-ink-per-detector-line'
    widths=[]
    for a,b,c,d in region.get('source_lines',[]):
        thickness=(c-a if vertical else d-b)-4
        length=d-b if vertical else c-a
        if 8<=thickness<=140 and length>=thickness*1.15:widths.append(thickness)
    if widths:return round(float(np.median(widths))),'full-source-ink-lines'
    lines=region.get('source_lines',[])
    if len(lines)==1:
        # One shouted line of outlined lettering (名まで推しを愛してる…, いーちっ) is wider than
        # the 140 px cap, and its glyph pieces fragment. A single column has no internal
        # gutter; merged columns do, and keep the cautious estimate below.
        a,b,c,d=lines[0];x0,y0=region['box'][:2]
        thickness=(c-a if vertical else d-b)-4;length=d-b if vertical else c-a
        local=np.asarray(mask)[y0+b:y0+d,x0+a:x0+c]>96
        across=local.any(0 if vertical else 1) if local.size else np.zeros(0,bool)
        filled=np.flatnonzero(across)
        solid=len(filled) and not (np.diff(filled)>3).any()
        if 140<thickness<=320 and length>=2.5*thickness and solid:
            return round(thickness*.8),'single-large-line'
    from paragraphs import glyph_height
    return round(glyph_height(mask.crop(region['box']))),'paragraph-glyphs'

def reconcile_regions(regions, mask, line_mask):
    if line_mask is None:return regions
    ink=np.asarray(mask)>96;lines=np.asarray(line_mask)>76
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    owned=[[] for _ in regions]
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<8:continue
        component=labels[y:y+h,x:x+w]==label
        if (lines[y:y+h,x:x+w]&component).sum()<area*.15:continue
        candidates=[];votes=[]
        for i,r in enumerate(regions):
            a,b,c,d=map(int,r['box'])
            ix,iy=max(a,x),max(b,y);iX,iY=min(c,x+w),min(d,y+h)
            if ix>=iX or iy>=iY:continue
            fraction=float((labels[iy:iY,ix:iX]==label).sum())/area
            if fraction<.35:continue
            votes.append(fraction)
            # Never absorb a page-sized/connected decorative stroke into a caption.
            limit=max(8,min(60,min(c-a,d-b)*.5))
            if max(a-x,b-y,x+w-c,y+h-d)>limit:continue
            candidates.append((fraction,i))
        candidates.sort(reverse=True)
        votes.sort(reverse=True)
        if len(votes)>1 and votes[0]-votes[1]<.15:continue
        if not candidates or (len(candidates)>1 and candidates[0][0]-candidates[1][0]<.15):continue
        owned[candidates[0][1]].append([int(x),int(y),int(x+w),int(y+h)])
    height,width=ink.shape
    for r,glyphs in zip(regions,owned):
        original=list(r['box']);a,b,c,d=original
        for x,y,X,Y in glyphs:a=min(a,x);b=min(b,y);c=max(c,X);d=max(d,Y)
        r['detector_box']=original
        r['box']=[max(0,a),max(0,b),min(width,c),min(height,d)]
        r['paragraph_geometry']={'source_box':r['box'][:],'glyph_components':glyphs,
            'anchor':[(a+c)/2,(b+d)/2],'method':'ctd-line-supported-glyphs',
            'expanded':r['box']!=original}
    # Direction-neutral DB components recover entire vertical columns, including
    # missed endpoints. Never attach a line that overlaps two competing paragraphs.
    contours,_=cv2.findContours(lines.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for poly in contours:
        x,y,w,h=cv2.boundingRect(poly)
        if min(w,h)<4 or w*h<64:continue
        support=ink[y:y+h,x:x+w]
        if support.mean()<.06:continue
        votes=[]
        for i,r in enumerate(regions):
            a,b,c,d=r['box'];inter=max(0,min(c,x+w)-max(a,x))*max(0,min(d,y+h)-max(b,y))
            if inter/(w*h)>.35:votes.append(i)
        if len(votes)!=1:continue
        r=regions[votes[0]];a,b,c,d=r['box'];limit=max(10,min(80,min(w,h)*1.5))
        if max(a-x,b-y,x+w-c,y+h-d)>limit:continue
        r['box']=[min(a,x),min(b,y),max(c,x+w),max(d,y+h)]
        g=r['paragraph_geometry'];g.setdefault('det_lines',[]).append([x,y,x+w,y+h])
        g['source_box']=r['box'][:];g['anchor']=[(r['box'][0]+r['box'][2])/2,(r['box'][1]+r['box'][3])/2]
        g['expanded']=r['box']!=r['detector_box']
    return regions

class WhiteEnclosures:
    """Which enclosed white area (balloon / caption box interior) holds a text box."""
    def __init__(self,image):
        import cv2
        rgb=np.asarray(image.convert('RGB'))
        white=((rgb.min(2)>218)&(np.ptp(rgb,axis=2)<28)).astype(np.uint8)
        _,self.labels,self.stats,_=cv2.connectedComponentsWithStats(white,8)
        self.page=image.width*image.height;self.size=image.size;self.gray=np.asarray(image.convert('L'))
    def dark(self,box):
        x1,y1,x2,y2=map(int,box);local=self.gray[y1:y2,x1:x2]
        return bool(local.size) and float(np.median(local))<90
    def of(self,box):
        x1,y1,x2,y2=map(int,box);w=max(1,x2-x1);h=max(1,y2-y1)
        ids,counts=np.unique(self.labels[y1:y2,x1:x2],return_counts=True)
        for i in np.argsort(-counts):
            label=int(ids[i])
            if not label or counts[i]<w*h*.15:continue
            x,y,bw,bh,area=map(int,self.stats[label])
            edges=sum((x<=1,y<=1,x+bw>=self.size[0]-1,y+bh>=self.size[1]-1))
            # Page margins and wide-open backgrounds are not an enclosure.
            if edges>=2 or area>self.page*.12:continue
            if x<=x1+2 and y<=y1+2 and x+bw>=x2-2 and y+bh>=y2-2:return label
        return None

def split_by_enclosure(r,mask,enclosures):
    """A detector block whose columns sit in two different balloons / caption boxes
    becomes one region per enclosure, so each bubble gets its own translation."""
    from direction import set_direction
    set_direction(r,mask,'ja')
    lines=r.get('source_lines',[])
    if len(lines)<2:return [r]
    x0,y0=r['box'][:2];keyed={}
    for a,b,c,d in lines:
        box=[x0+a,y0+b,x0+c,y0+d];keyed.setdefault(enclosures.of(box),[]).append(box)
    labelled={k:v for k,v in keyed.items() if k is not None};stray=keyed.get(None,[])
    if len(labelled)==1 and stray and all(enclosures.dark(b) and (b[2]-b[0])*(b[3]-b[1])>=900 for b in stray):
        # Large white lettering on a black panel next to a balloon (名まで推しを愛してる…)
        # is its own text, never a column of that balloon.
        labelled['dark']=stray;stray=[]
    if len(labelled)<2:return [r]
    # Stray lines on an outline join the nearest enclosure's group.
    for box in stray:
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        nearest=min(labelled,key=lambda k:min(abs((b[0]+b[2])/2-cx)+abs((b[1]+b[3])/2-cy) for b in labelled[k]))
        labelled[nearest].append(box)
    pieces=[]
    for boxes in labelled.values():
        box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        piece=dict(r);piece['box']=box;piece['balloon_split']=True
        piece['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],'method':'split-by-enclosure'}
        pieces.append(piece)
    return pieces

def ink_columns(r,mask):
    """Vertical ink runs inside a region box: (left, right, top, bottom) in page pixels."""
    x1,y1,x2,y2=map(int,r['box'])
    ink=np.asarray(mask)[y1:y2,x1:x2]>96
    if not ink.size:return []
    profile=np.r_[ink.any(0),False];runs=[];start=None
    for x,v in enumerate(profile):
        if v and start is None:start=x
        elif not v and start is not None:
            rows=np.flatnonzero(ink[:,start:x].any(1))
            runs.append((x1+start,x1+x,y1+int(rows[0]),y1+int(rows[-1])+1));start=None
    return runs

def outline_above(gray,group,limit):
    """Median distance from a column group's top up to the first dark outline pixel."""
    left=min(c[0] for c in group);right=max(c[1] for c in group);top=min(c[2] for c in group)
    span=right-left;found=[]
    for x in range(int(left+span*.2),int(right-span*.2)+1,max(1,span//12)):
        col=gray[max(0,limit):top,x][::-1]
        hit=np.flatnonzero(col<110)
        found.append(int(hit[0])+1 if len(hit) else top-limit)
    return float(np.median(found)) if found else float(top-limit)

def outline_below(gray,group,limit):
    left=min(c[0] for c in group);right=max(c[1] for c in group);bottom=max(c[3] for c in group)
    span=right-left;found=[]
    for x in range(int(left+span*.2),int(right-span*.2)+1,max(1,span//12)):
        hit=np.flatnonzero(gray[bottom:min(gray.shape[0],limit),x]<110)
        found.append(int(hit[0])+1 if len(hit) else limit-bottom)
    return float(np.median(found)) if found else float(limit-bottom)

def split_staggered(r,mask,gray=None):
    """Two caption boxes drawn as one joined outline (an L or step shape) form a single
    white area, so the enclosure test cannot part them. Their column groups still show
    it: going right to left, the second box's columns start AND end clearly lower (or
    higher) after a wider gutter. Each group becomes its own paragraph."""
    if r.get('orientation')!='vertical':return [r]
    runs=ink_columns(r,mask)
    if len(runs)<2:return [r]
    glyph=float(r.get('source_font_size') or 0) or float(np.median([b-a for a,b,_,_ in runs]))
    wide=max(b-a for a,b,_,_ in runs)
    # Ruby and dashes are narrow runs; they follow whichever column they sit beside.
    main=sorted([c for c in runs if c[1]-c[0]>=wide*.55],key=lambda c:-c[0])
    if len(main)<2:return [r]
    groups=[[main[0]]];step=glyph*1.5
    for prev,col in zip(main,main[1:]):
        gutter=prev[0]-col[1];dt=col[2]-prev[2];db=col[3]-prev[3]
        if gutter>=glyph*.8 and ((dt>step and db>step) or (dt<-step and db<-step)):groups.append([col])
        else:groups[-1].append(col)
    if gray is not None and len(groups)>1:
        # Staggered columns inside ONE balloon (譲りま / しょうか…？) leave open balloon
        # interior on one side; in a joined two-box outline the lower box's own edge sits
        # right above the lower group AND the upper box's edge right below the upper one.
        kept=[groups[0]]
        for g in groups[1:]:
            prev=kept[-1];upper,lower=(prev,g) if min(c[2] for c in g)>min(c[2] for c in prev) else (g,prev)
            up_top=min(c[2] for c in upper);low_bottom=max(c[3] for c in lower)
            above=outline_above(gray,lower,up_top);below=outline_below(gray,upper,low_bottom)
            # Measured: joined boxes 1.1/0.8, 1.4/1.0, 1.8/1.5 glyphs; one balloon 1.4/3.3.
            if above<glyph*2 and below<glyph*2:kept.append(g)
            else:prev.extend(g)
        if len(groups)==2 and len(kept)==1:
            # One joined balloon holding a high right group and a low left group: keep one
            # paragraph, but remember both lobes so the Chinese is set the same way.
            r['stagger_lobes']=[[min(c[0] for c in g),min(c[2] for c in g),max(c[1] for c in g),max(c[3] for c in g)] for g in groups]
        groups=kept
    if len(groups)<2:return [r]
    for c in runs:
        if c in main:continue
        cx=(c[0]+c[1])/2
        min(groups,key=lambda g:min(abs((a+b)/2-cx) for a,b,_,_ in g)).append(c)
    pieces=[];lines=r.get('paragraph_geometry',{}).get('det_lines',[])
    for g in groups:
        box=[min(c[0] for c in g),min(c[2] for c in g),max(c[1] for c in g),max(c[3] for c in g)]
        piece=dict(r);piece['box']=box;piece['balloon_split']=True;piece.pop('paragraph_parts',None)
        own=[l for l in lines if box[0]<=(l[0]+l[2])/2<=box[2] and box[1]<=(l[1]+l[3])/2<=box[3]]
        piece['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],'method':'split-staggered-columns','det_lines':own}
        pieces.append(piece)
    return pieces

def split_by_background(r,mask,gray):
    """White lettering on a black panel and black lettering in the white balloon beside
    it (名まで推しを愛してる… / と言います！) can share one detector box. A narrow balloon
    column then looks like ruby to split_staggered. Paper tone tells them apart."""
    if r.get('orientation')!='vertical' or gray is None:return [r],None
    runs=ink_columns(r,mask)
    if len(runs)<2:return [r],None
    ink=np.asarray(mask)>96
    def tone(c):
        a,b,t,u=c;paper=gray[t:u,a:b][~ink[t:u,a:b]]
        return float(np.median(paper)) if paper.size>50 else 128.
    tones=[tone(c) for c in runs]
    dark=[c for c,t in zip(runs,tones) if t<80];light=[c for c,t in zip(runs,tones) if t>170]
    if not dark or not light or len(dark)+len(light)<len(runs):return [r],None
    pieces=[];lines=r.get('paragraph_geometry',{}).get('det_lines',[])
    for g in (dark,light):
        box=[min(c[0] for c in g),min(c[2] for c in g),max(c[1] for c in g),max(c[3] for c in g)]
        piece=dict(r);piece['box']=box;piece['balloon_split']=True;piece.pop('paragraph_parts',None)
        own=[l for l in lines if box[0]<=(l[0]+l[2])/2<=box[2] and box[1]<=(l[1]+l[3])/2<=box[3]]
        piece['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],'method':'split-by-background','det_lines':own}
        pieces.append(piece)
    return pieces,pieces[1]

def join_balloon_neighbour(piece,result,enclosures):
    """The light piece split off above is usually the last column of the balloon text
    beside it (押尾愛 / と言います！): join that paragraph instead of standing alone."""
    label=enclosures.of(piece['box'])
    if label is None:return False
    x1,y1,x2,y2=piece['box'];w=x2-x1
    for other in result:
        if other is piece or enclosures.of(other['box'])!=label:continue
        a,b,c,d=other['box'];gap=max(a-x2,x1-c,0);overlap=min(y2,d)-max(y1,b)
        if gap<=w*1.5 and overlap>.5*(y2-y1):
            other['box']=[min(a,x1),min(b,y1),max(c,x2),max(d,y2)]
            geometry=other.setdefault('paragraph_geometry',{})
            geometry['det_lines']=list(geometry.get('det_lines',[]))+piece['paragraph_geometry']['det_lines']
            geometry['source_box']=other['box'][:]
            return True
    return False

def ink_tone(rgb,mask,box):
    """Median colour of the darker glyph pixels: printed lettering is solid black, pen or
    pencil notes (簡単そうなのに / せんぱいが用意した浴衣) are anti-aliased gray or brown."""
    x1,y1,x2,y2=map(int,box)
    px=rgb[y1:y2,x1:x2][np.asarray(mask)[y1:y2,x1:x2]>64]
    px=px[px.mean(1)<170] if len(px) else px
    if len(px)<20:return None
    med=np.median(px,0);return float(med.mean()),float(np.ptp(med))

def pen_note(tone):
    return tone is not None and tone[0]>=33 and tone[1]<=25

def split_pen_columns(r,mask,rgb):
    """A pen note written beside a balloon joins its detector box; part it from the
    printed columns so it can be left as drawn."""
    if r.get('orientation')!='vertical':return [r]
    runs=[c for c in ink_columns(r,mask) if c[3]-c[2]>=1.2*(c[1]-c[0]) and c[1]-c[0]>=12]
    if len(runs)<2:return [r]
    tones=[ink_tone(rgb,mask,(a,t,b,u)) for a,b,t,u in runs]
    pen=[c for c,t in zip(runs,tones) if pen_note(t)];printed=[c for c,t in zip(runs,tones) if t is not None and t[0]<=20]
    if not pen or not printed:return [r]
    # Ellipsis dots and slivers are not a note column.
    pen=[c for c in pen if c[1]-c[0]>=.5*min(p[1]-p[0] for p in printed)]
    if not pen:return [r]
    pieces=[];lines=r.get('paragraph_geometry',{}).get('det_lines',[])
    for g,is_pen in ((printed,False),(pen,True)):
        box=[min(c[0] for c in g),min(c[2] for c in g),max(c[1] for c in g),max(c[3] for c in g)]
        piece=dict(r);piece['box']=box;piece['balloon_split']=True;piece.pop('paragraph_parts',None)
        if is_pen:piece['pen_note']=True
        own=[l for l in lines if box[0]<=(l[0]+l[2])/2<=box[2] and box[1]<=(l[1]+l[3])/2<=box[3]]
        piece['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],'method':'split-pen-columns','det_lines':own}
        pieces.append(piece)
    return pieces

def merge_japanese(image,mask,regions):
    """Join CTD pieces only with independent evidence of the SAME text paragraph."""
    from visual_evidence import coverage
    from direction import set_direction
    from paragraphs import glyph_height
    groups={};result=[]
    for r in regions:
        evidence=r.get('visual_evidence',{})
        if evidence.get('class') in (1,2) and 'text_id' in evidence:
            groups.setdefault(evidence['text_id'],[]).append(r)
        else:result.append(r)
    enclosures=None
    split=[]
    for parts in groups.values():
        if len(parts)>1:
            # One detector text box can span two touching balloons or caption boxes
            # (interlocking hexagons, staggered narration boxes). Pieces that sit in
            # different enclosed white areas are separate paragraphs.
            enclosures=enclosures or WhiteEnclosures(image)
            keyed={}
            for p in parts:keyed.setdefault(enclosures.of(p['box']),[]).append(p)
            if len([k for k in keyed if k is not None])>1:
                for k,members in keyed.items():
                    if k is None:result.extend(members)
                    else:split.append((members,False))
                continue
        split.append((parts,True))
    for parts,whole in split:
        if not whole:
            if len(parts)==1:result.append(parts[0]);continue
            boxes=[r['box'] for r in parts]
            box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            r=dict(parts[0]);r['box']=box;r['paragraph_parts']=boxes;r['balloon_split']=True
            r['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],'method':'ctd-lines-per-balloon',
                'glyph_components':[g for p in parts for g in p.get('paragraph_geometry',{}).get('glyph_components',[])],
                'det_lines':[line for p in parts for line in p.get('paragraph_geometry',{}).get('det_lines',[])]}
            result.append(r);continue
        text_box=parts[0]['visual_evidence']['text_box'];boxes=[r['box'] for r in parts]
        box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        if coverage(box,text_box)<.8:result.extend(parts);continue
        # The independent text head can see an endpoint that the glyph head missed.
        # It is a text rectangle, never the enclosing speech balloon rectangle.
        if parts[0]['visual_evidence'].get('score',0)>=.7:
            box=[min(box[0],text_box[0]),min(box[1],text_box[1]),max(box[2],text_box[2]),max(box[3],text_box[3])]
        r=dict(parts[0]);r['box']=box;r['paragraph_parts']=boxes
        r['paragraph_geometry']={'source_box':box[:],'anchor':[(box[0]+box[2])/2,(box[1]+box[3])/2],
            'method':'ctd-lines-rtdetr-paragraph','glyph_components':[g for p in parts for g in p.get('paragraph_geometry',{}).get('glyph_components',[])],
            'det_lines':[line for p in parts for line in p.get('paragraph_geometry',{}).get('det_lines',[])]}
        result.append(r)
    enclosures=enclosures or WhiteEnclosures(image)
    result=[piece for r in result for piece in split_by_enclosure(r,mask,enclosures)]
    for r in result:
        if 'source_font_size' not in r:r['source_font_size'],_=source_ink_size(r,mask)
    gray=np.asarray(image.convert('L'))
    result=[piece for r in result for piece in split_staggered(r,mask,gray)]
    loose=[]
    for r in list(result):
        pieces,light=split_by_background(r,mask,gray)
        if light is None:continue
        i=result.index(r);result[i:i+1]=pieces;loose.append(light)
    for piece in loose:
        if join_balloon_neighbour(piece,result,enclosures):result.remove(piece)
    rgb=np.asarray(image.convert('RGB')).astype(int)
    result=[piece for r in result for piece in split_pen_columns(r,mask,rgb)]
    for r in result:r.pop('source_font_size',None)
    for r in result:
        set_direction(r,mask,'ja')
        r['source_font_size'],r['source_font_size_method']=source_ink_size(r,mask)
        r.setdefault('paragraph_geometry',{}).update(source_lines=r['source_lines'],orientation=r['orientation'],font_size=r['source_font_size'])
        if r['orientation']=='vertical' and enclosures.of(r['box']) is None and pen_note(ink_tone(rgb,mask,r['box'])):r['pen_note']=True
    result.sort(key=lambda r:(r['box'][1]//180,-r['box'][0]))
    for i,r in enumerate(result):r['id']=i
    return result
