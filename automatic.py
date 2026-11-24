"""One-folder workflow: automatic recovery, original names, clean sibling output."""
import hashlib,json,os,shutil,re,collections
from pathlib import Path
from PIL import Image,ImageOps
import engine
from text_rules import normalize_preserved,preserve,needs_review,QUIET_STATUSES,punctuation_only,vocalization_only
from text_style import recover_flat_mask
from paragraphs import merge_paragraphs
from task_storage import locked_run,commit_page

CACHE_ROOT=engine.ROOT/'runtime/jobs'

def job_directory(source,output):
    key=hashlib.sha256((str(Path(source).resolve()).casefold()+'\0'+str(Path(output).resolve()).casefold()).encode()).hexdigest()[:24]
    return CACHE_ROOT/key

def read_manifest(source,output):
    try:
        data=json.loads((job_directory(source,output)/'manifest.json').read_text(encoding='utf-8'))
        if data['source']==str(Path(source).resolve()) and data['output']==str(Path(output).resolve()):return data
    except (OSError,ValueError,KeyError):pass
    return None

def sibling_output(source):
    source=Path(source).resolve()
    for number in range(1,10000):
        suffix='（中译）' if number==1 else f'（中译 {number}）'
        output=source.with_name(source.name+suffix)
        if output.is_symlink() or output.is_junction():continue
        if not output.exists() or (output.is_dir() and not any(output.iterdir())) or read_manifest(source,output):return output
    raise ValueError('找不到可用的同级输出目录。')

def scan_images(source):
    source=Path(source).resolve();found=[]
    for parent,dirs,files in os.walk(source,followlinks=False):
        dirs[:]=[n for n in dirs if n!='.passage' and not (Path(parent)/n).is_symlink() and not (Path(parent)/n).is_junction()]
        for name in files:
            p=Path(parent)/name
            if p.suffix.lower() in engine.EXTENSIONS and not p.is_symlink() and source in p.resolve().parents:found.append(p)
    return sorted(found,key=lambda p:[int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)',str(p.relative_to(source)))])

def checked_path(root,path):
    root=Path(root).resolve();path=Path(path)
    if root not in path.resolve().parents:raise ValueError('目标路径超出输出目录。')
    for item in (path,*path.parents):
        if item==root:break
        if item.is_symlink() or item.is_junction():raise ValueError('目标路径包含链接，已停止写入。')
    return path

def remove_working(job):
    job=Path(job).resolve();cache=CACHE_ROOT.resolve();target=job/'working'
    if cache not in job.parents or target.resolve().parent!=job or target.is_symlink() or target.is_junction():
        raise ValueError('工作目录校验失败，未清理。')
    if target.exists():shutil.rmtree(target)

def save_matching(image,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.passage-tmp')
    formats={'.jpg':'JPEG','.jpeg':'JPEG','.png':'PNG','.webp':'WEBP','.bmp':'BMP','.tif':'TIFF','.tiff':'TIFF','.avif':'AVIF','.gif':'GIF'}
    fmt=formats[target.suffix.lower()];options={}
    if fmt=='JPEG':image=image.convert('RGB');options={'quality':96,'subsampling':0}
    elif fmt=='WEBP':options={'lossless':True,'method':4}
    elif fmt=='TIFF':options={'compression':'tiff_deflate'}
    elif fmt=='AVIF':options={'quality':95}
    try:
        image.save(temp,format=fmt,**options)
        with Image.open(temp) as check:check.load();assert check.size==image.size
        temp.replace(target)
    finally:
        if temp.exists():temp.unlink()

def copy_atomic(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.passage-tmp')
    try:shutil.copyfile(source,temp);temp.replace(target)
    finally:
        if temp.exists():temp.unlink()

def auto_preserve(r,reason):
    r['automatic_reason']=reason;preserve(r,'自动保留原文');r.pop('layout',None)

def expected_preservation(r):
    return not r.get('automatic_reason') and (r.get('kind')=='sfx' or r.get('status') in {
        '跳过标点，保留原文','短字或符号，保留原文','短句无法可靠翻译，保留原文','无需翻译，保留原文'})

def unresolved_count(regions):
    return sum(bool(r.get('automatic_reason')) or r.get('post_render_verified') is False or
               (r.get('status')!='已嵌字' and not expected_preservation(r)) for r in regions)

def classify_nonprose(image,regions,r):
    """After OCR recovery, separate short effects/drawing detections from prose."""
    from typesetting import BubbleMap
    if not r.get('automatic_ocr_retry'):return False
    lines=r.get('ocr_lines',[]);scores=r.get('ocr_line_scores',[])
    reliable_words=any(score>=.8 and re.search('[A-Za-z]{3,}',line) for line,score in zip(lines,scores))
    if reliable_words:return False
    a,b,c,d=r['box'];count=r.get('ocr_segmentation',{}).get('line_count',0)
    short_shape=(0<count<=2 or 0<count<=6 and r.get('source_font_size',0)*.8/max(1,c-a)>=.3 or count==0 and max(c-a,d-b)<200)
    if short_shape and len(re.findall('[A-Za-z]',r.get('source','')))<=4:
        preserve(r,'短句无法可靠翻译，保留原文');r.pop('automatic_reason',None);return True
    if BubbleMap(image,regions).find(r) is None:
        r.update(kind='sfx',translation='',status='跳过不规则文字',skip_review=True,
                 automatic_classification='两次识别均未得到可靠英文词，位于对话框外的装饰字或非文字形状')
        r.pop('automatic_reason',None);return True
    return False

def in_balloon(r):
    """A drawn balloon or boxed caption: RT-DETR text-in-bubble or bubble seed, or a
    bordered balloon confirmed by the contour check."""
    v=r.get('visual_evidence') or {}
    return bool(r.get('boxed_voice') or r.get('boxed_prose') or v.get('class')==1 or v.get('bubble_box'))

# Fillers and set phrases only: OCR of a sign or a sound effect easily yields それは / これ.
SPOKEN_WORDS={'あの','あのう','あのね','えっと','えと','ねえ','ねぇ','はい','うん','ううん','いいえ',
              'ほら','なんで','どうも','だって','まあ','よし','さあ','おい',
              'ちょっと','ごめん','ごめんね','どうぞ','すみません','ありがとう'}

def outside_balloon_effect(r,unreadable=False):
    """User rule (2026-11-16): cries and sound effects without a balloon are never
    translated or erased; erasing them damages the artwork drawn around them."""
    if in_balloon(r):return False
    text=re.sub(r'[\W_ー〜～]','',r.get('source',''))
    if vocalization_only(r.get('source','')):return True
    if (r.get('visual_evidence') or {}).get('class')==2:return False  # detector: plain narration text
    if unreadable:return len(text)<=6
    # Short everyday words (\u3042\u306e / \u3048\u3063\u3068 / \u307b\u3089) are speech, not sound effects, even when
    # their balloon lobe is not recognised; only cries and effects stay untouched.
    if text in SPOKEN_WORDS:return False
    return bool(text) and len(text)<=5 and bool(re.fullmatch(r'[\u3040-\u30ff]+',text))

def mark_effect(r):
    r.update(kind='sfx',translation='',status='跳过特效字',skip_review=True,
             automatic_classification='对话框外的叫声或拟声字，按规则保留原图')
    r.pop('automatic_reason',None);r.pop('layout',None)

def cover_candidate(image,vision):
    """Only high-confidence single-illustration color covers skip automatically.

    Monochrome/ambiguous first pages continue through normal translation.
    """
    import numpy as np
    from typesetting import BubbleMap
    rgb=np.asarray(image.resize((320,320)).convert('RGB')).astype(int)
    colorful=float((np.ptp(rgb,axis=2)>35).mean())
    if colorful<.22:return False
    regions,_=vision.detect(image);bubbles=BubbleMap(image,regions)
    enclosed=sum(bool(bubbles.find(r)) for r in regions)
    large=sum((r['box'][3]-r['box'][1]>.16*image.height or r['box'][2]-r['box'][0]>.3*image.width) for r in regions)
    return enclosed<=1 and large>=2

class AutoBatch:
    def __init__(self,source,output=None,language='ja',glossary='',log=print,event=None,cancel=None,reprocess=False,audience='auto'):
        self.source,self.output=engine.validate_roots(source,output or sibling_output(source))
        self.language=language;self.glossary=glossary;self.log=log;self.event=event or (lambda *a:None);self.cancel=cancel or (lambda:False)
        self.reprocess=reprocess;self.history=[]
        self.vision=engine.Vision();self.translator=engine.LocalTranslator(log)
        # One automatic register: everyday tone always, adult glossary only where the source uses those words.
        self.audience=audience if audience in ('auto','general','adult') else 'auto';self.translator.audience=self.audience
        self.coverage=None
        self.job=job_directory(self.source,self.output)
        self.settings=hashlib.sha256(('auto-1.4.0-r4'+language+glossary+('|general' if self.audience=='general' else '')).encode()).hexdigest()
        self.state=read_manifest(self.source,self.output) or {'source':str(self.source),'output':str(self.output),'pages':{},'companions':{}}
        self.summary={'completed':0,'resumed':0,'failed':0,'cancelled':False,'review':0,'retained_pages':0,'preserved_regions':0,'intentional_skips':0,'covers':0}

    def check(self):
        if self.cancel():raise engine.Cancelled()

    def checkpoint(self,status='running'):
        self.state.update(status=status,version=engine.VERSION,summary=self.summary)
        engine.atomic_json(self.job/'manifest.json',self.state)

    def retry_ocr(self,image,mask,r,vision=None):
        vision=vision or self.vision
        box=r['box'];crop=image.crop(box);cmask=mask.crop(box)
        recovered=recover_flat_mask(image,box)
        if recovered is not None:cmask=recovered
        size=(max(1,crop.width*2),max(1,crop.height*2))
        temp={'id':r['id'],'box':[0,0,*size],'source':'','translation':'','kind':'dialogue','status':'待识别'}
        ok=vision.recognize(crop.resize(size,Image.Resampling.LANCZOS),cmask.resize(size,Image.Resampling.NEAREST),temp,self.language)
        r['automatic_ocr_retry']=True
        if ok:
            for key in ('source','ocr_confidence','kind','status','ocr_lines','ocr_line_scores','ocr_segmentation'):r[key]=temp.get(key)
            r.pop('skip_review',None);r.pop('automatic_reason',None)
            if recovered is not None:mask.paste(recovered,box[:2])
        return ok

    def translate_group(self,group):
        try:self.translator.translate(group,self.language,self.glossary,getattr(self,'history',[])[-8:])
        except (ValueError,json.JSONDecodeError):pass
        except RuntimeError as exc:
            if '有效结果' not in str(exc):raise
        for r in group:
            self.check()
            prose=len(re.findall(r"[A-Za-z]+",r.get('source','')))>=3
            retry=not r.get('translation') and (r.get('kind') not in ('sfx','preserve') or self.language=='en' and prose)
            if retry:
                r.update(kind='dialogue',status='待翻译');r.pop('skip_review',None)
                try:
                    if hasattr(self.translator,'retry_translation'):self.translator.retry_translation(r,self.language,self.glossary)
                    else:self.translator.translate([r],self.language,self.glossary,[])
                except (ValueError,json.JSONDecodeError):pass
                except RuntimeError as exc:
                    if '有效结果' not in str(exc):raise
            if retry and not r.get('translation'):
                auto_preserve(r,'翻译重试后仍无法可靠确定')
        if self.language=='ja' and hasattr(self.translator,'review_translations'):
            self.log('对照原文校核增译、遗漏与语气…')
            self.translator.review_translations(group,self.language,self.glossary)
            for r in group:
                if r.get('semantic_review')=='unresolved':auto_preserve(r,'原文对照校核未通过')
        self.history=(getattr(self,'history',[])+[{'source':r.get('source',''),'translation':r['translation']} for r in group if r.get('translation')])[-8:]

    def process(self,image,prior=None):
        return self.finish(image,*self.analyze(image,prior))

    def analyze(self,image,prior=None,vision=None,log=None):
        """Detection, OCR and effect/dialogue classification. CPU only and no translator
        use, so the next page can be analysed while this page is translated and lettered."""
        vision=vision or self.vision;log=log or self.log
        if self.language=='en' and hasattr(self.vision,'ocr'):
            if self.coverage is None:
                from english_coverage import EnglishCoverage
                self.coverage=EnglishCoverage(engine.ROOT)
            regions,mask=self.coverage.detect(image)
            # CTD remains the independent stroke mask / second OCR source.
            _,ctd_mask=self.vision.detect(image)
            import numpy as np
            mask=Image.fromarray(np.maximum(np.asarray(mask),np.asarray(ctd_mask)))
        else:regions,mask=vision.detect(image)
        if not regions:
            log('重新增强对比度检测文字…');regions,mask=vision.detect(ImageOps.autocontrast(image))
        if self.language!='ja' and hasattr(self.vision,'last_evidence_hits'):
            from visual_evidence import annotate
            annotate(regions,self.vision.last_evidence_hits,mask,getattr(self.vision,'last_line_mask',None))
        if not regions or regions[0].get('detector')!='PP-OCRv6 full-page':regions=merge_paragraphs(image,mask,regions,self.language)
        if self.language!='ja':
            regions.sort(key=lambda r:(r['box'][1]//180,r['box'][0]))
            for i,r in enumerate(regions):r['id']=i
        active=[];balloons=None
        for r in regions:
            self.check()
            if r.get('detector')=='PP-OCRv6 full-page':
                ok=r.get('kind')!='sfx'
                if ok:
                    original_text=r['source'];alternative=dict(r);alternative['kind']='dialogue'
                    read=self.vision.recognize(image,mask,alternative,'en')
                    old_length=len(re.sub(r'\W','',original_text));new_length=len(re.sub(r'\W','',alternative.get('source','')))
                    if read and alternative.get('ocr_confidence',0)>.88 and .8*old_length<=new_length<=1.4*old_length:
                        r.update(alternative);r['independent_ocr_agreement']=True
            else:ok=vision.recognize(image,mask,r,self.language)
            if self.language=='ja' and not ok and r.get('status')=='识别不确定，保留原文' and outside_balloon_effect(r,unreadable=True):
                # Unreadable art outside any balloon: no second OCR pass, left as drawn.
                mark_effect(r);continue
            if not ok and r.get('status')=='识别不确定，保留原文':
                log('自动复核低置信度识别区域…');ok=self.retry_ocr(image,mask,r,vision)
            from text_rules import mark_boxed_voice
            if mark_boxed_voice(image,regions,r) and not r.get('ocr_repetition_stopped') and r.get('status')!='识别不确定，保留原文':
                ok=True;r['status']='待翻译'
            if self.language=='ja':
                if ok and outside_balloon_effect(r):
                    # The detector can miss a small balloon (あの / …！やっぱ…); a closed,
                    # outlined white balloon around the text still makes it dialogue.
                    if balloons is None:
                        from typesetting import BubbleMap
                        balloons=BubbleMap(image,regions)
                    found=balloons.find(r);x1,y1,x2,y2=r['box']
                    snug=found and found['box'][2]-found['box'][0]<=6*(x2-x1) and found['box'][3]-found['box'][1]<=4*(y2-y1)
                    if not (snug and found.get('bordered')):mark_effect(r);continue
                if in_balloon(r) and vocalization_only(r.get('source','')) and not r.get('ocr_repetition_stopped') and r.get('status')!='识别不确定，保留原文':
                    # A clear cry inside a drawn balloon is always translated, never half-skipped.
                    r['boxed_voice']=True;r.update(kind='dialogue',status='待翻译');r.pop('skip_review',None);ok=True
            if not ok and self.language=='en' and r.get('status')=='识别不确定，保留原文':classify_nonprose(image,regions,r)
            if ok:
                cached=next((old for old in (prior or []) if (old.get('box')==r['box'] or len(re.sub(r'\W','',r.get('source','')))>12) and
                    re.sub(r'\W','',old.get('source','')).casefold()==re.sub(r'\W','',r.get('source','')).casefold() and old.get('translation') and old.get('status')=='已嵌字'),None)
                if cached and engine.invalid_translation(r['source'],cached['translation']):cached=None
                if cached and re.search(r'奶油|失败者|陷阱',cached.get('translation','')) and re.search(r'creampie|loser|trap',r.get('source',''),re.I):cached=None
                if cached:
                    r.update(translation=engine.normalize_romanized_names(r['source'],cached['translation']),kind=cached['kind'],status='已翻译',translation_reused=True)
                    if cached.get('translator'):r['translator']=cached['translator']
                else:active.append(r)
            elif not normalize_preserved(r) and r.get('kind')!='sfx' and r.get('status') not in QUIET_STATUSES:auto_preserve(r,'识别重试后仍不明确')
        return regions,mask,active

    def finish(self,image,regions,mask,active):
        """Translation, cleaning and lettering of an analysed page (main thread only)."""
        for start in range(0,len(active),4):
            self.check();self.log('本地翻译与自动纠正…');self.translate_group(active[start:start+4])
        # Retry missing masks only on verified regular dialogue, then rerender from originals.
        for r in regions:
            if r.get('translation') and mask.crop(r['box']).getbbox() is None:
                recovered=recover_flat_mask(image,r['box'])
                if recovered is not None:mask.paste(recovered,r['box'][:2])
        for attempt in range(3):
            self.check()
            try:result=engine.render(image,mask,regions,self.vision,self.language,self.log,automatic=True)
            except Exception as exc:
                self.log('该页清字无法可靠完成，自动保留原图：'+str(exc))
                for r in regions:
                    if r.get('translation'):auto_preserve(r,'清字失败，保留原始画面')
                return image.copy(),regions
            unresolved=[r for r in regions if r.get('translation') and r.get('status')!='已嵌字']
            if not unresolved:
                if self.coverage is not None:
                    hits=self.coverage.residuals(result,regions)
                    hit_ids={hit['id'] for hit in hits}
                    for r in regions:
                        if r.get('status')=='已嵌字':r['post_render_verified']=r['id'] not in hit_ids
                    if hits and attempt==0:
                        self.log('全页英文复核发现残留，自动重译或重新清字…')
                        for r in regions:
                            copied={w for hit in hits if hit['id']==r['id'] for w in hit['words']} & set(re.findall('[A-Za-z]{3,}',r.get('translation','').upper()))
                            if copied:
                                r['failed_translation']=r['translation']
                                self.translator.retry_translation(r,self.language,self.glossary)
                        mask=mask.filter(__import__('PIL.ImageFilter',fromlist=['MaxFilter']).MaxFilter(5));continue
                    for r in regions:
                        if r['id'] in hit_ids:r['automatic_reason']='全页复核仍检出原文'
                    return result,regions
                # Audit the written pixels, not merely the translation model's status.
                if self.language=='en' and attempt<=1:
                    from paragraphs import english_ink
                    import numpy as np
                    residual=[]
                    for r in regions:
                        if r.get('status')!='已嵌字':continue
                        words=set(re.findall(r'[A-Z]{4,}',r.get('source','').upper()))
                        if not words:continue
                        check={'id':r['id'],'box':r['box'],'source':'','translation':'','kind':'unknown'}
                        self.vision.recognize(result,mask,check,'en')
                        found=words & set(re.findall(r'[A-Z]{4,}',check.get('source','').upper()))
                        r['post_render_verified']=not bool(found)
                        if found:
                            residual.append(r)
                            ink=english_ink(image.crop(r['box']),mask.crop(r['box']))
                            cmask=np.maximum(np.asarray(mask.crop(r['box'])),ink.astype(np.uint8)*255)
                            mask.paste(Image.fromarray(cmask),r['box'][:2]);r['automatic_cleaning_retry']=True
                    if residual and attempt==0:
                        self.log('输出复核发现残留原文，自动重新清字与排版…');continue
                    if residual:
                        for r in residual:r['automatic_reason']='重新清字后仍检出与原文匹配的外语词'
                return result,regions
            for r in unresolved:auto_preserve(r,r.get('status','排版无法可靠完成'))
        # New preservation zones can invalidate neighboring layouts; preserve that page intact.
        for r in regions:
            if r.get('translation'):auto_preserve(r,'相邻文字区域冲突，保留原始画面')
        return image.copy(),regions

    def companions(self):
        for parent,dirs,files in os.walk(self.source,followlinks=False):
            dirs[:]=[name for name in dirs if name!='.passage' and not (Path(parent)/name).is_symlink() and not (Path(parent)/name).is_junction()]
            rel=Path(parent).relative_to(self.source)
            if rel.parts:checked_path(self.output,self.output/rel).mkdir(parents=True,exist_ok=True)
            for name in files:
                self.check();src=Path(parent)/name
                if src.suffix.lower() in engine.EXTENSIONS or src.is_symlink() or name=='.passage':continue
                dst=checked_path(self.output,self.output/rel/name);key=str(rel/name);digest=engine.sha256(src)
                old=self.state['companions'].get(key)
                if dst.exists() and engine.sha256(dst)!=digest and (not old or engine.sha256(dst)!=old['hash']):raise ValueError('已有同名非图片文件，未覆盖：'+key)
                if not dst.exists() or engine.sha256(dst)!=digest:copy_atomic(src,dst)
                self.state['companions'][key]={'hash':digest}

    @locked_run
    def run(self,files=None):
        missing=engine.model_status()
        if missing:raise RuntimeError('模型尚未就绪：'+', '.join(missing))
        engine.offline_guard();files=files if files is not None else scan_images(self.source)
        self.output.mkdir(parents=True,exist_ok=True);self.checkpoint()
        first_page=(scan_images(self.source) or [None])[0]
        from concurrent.futures import ThreadPoolExecutor
        pool=ThreadPoolExecutor(1,thread_name_prefix='passage-ahead') if self.language=='ja' else None
        ahead={}
        def prefetch(index):
            if pool is None or index>=len(files):return
            nxt=Path(files[index])
            if str(nxt) in ahead or nxt==first_page:return
            if not self.reprocess and checked_path(self.output,self.output/nxt.relative_to(self.source)).exists():return
            ahead[str(nxt)]=pool.submit(self.analyze_file,nxt)
        try:
            for i,path in enumerate(files):
                self.check();path=Path(path);rel=path.relative_to(self.source);key=str(rel);dst=checked_path(self.output,self.output/rel)
                self.log(f'[{i+1}/{len(files)}] {rel}');self.event('page',str(path),str(dst),i,len(files))
                try:
                    digest=engine.sha256(path);old=self.state['pages'].get(key)
                    if dst.exists():
                        current=engine.sha256(dst)
                        if not old or old.get('output_sha256')!=current:raise ValueError('已有文件不属于可验证的本次结果，未覆盖。')
                        if not self.reprocess and old.get('source_sha256')!=digest:raise ValueError('原图发生变化，已保存页面未覆盖；请明确选择重做此页。')
                        if old.get('source_sha256')==digest and not self.reprocess:
                            self.summary['resumed']+=1;self.summary['retained_pages']+=int(old.get('retained',False));self.summary['preserved_regions']+=old.get('preserved_regions',0)
                            self.summary['intentional_skips']+=old.get('intentional_skips',0);self.summary['covers']+=int(old.get('note')=='封面海报原样保留')
                            meta=self.job/'working'/rel.parent/(rel.name+'.json')
                            try:
                                previous=json.loads(meta.read_text(encoding='utf-8')).get('regions',[])
                                context=[{'source':r.get('source',''),'translation':r['translation']} for r in previous if r.get('status')=='已嵌字' and r.get('translation')]
                                self.history=(self.history+context)[-8:]
                                if hasattr(self.translator,'sakura_history'):self.translator.sakura_history=(self.translator.sakura_history+[r['translation'] for r in context])[-8:]
                            except (OSError,ValueError):pass
                            self.event('done',str(path),str(dst),'已保存（有未译文字）' if old.get('retained') else '完成（已保存）',str(meta) if meta.exists() else '');continue
                    prefetch(i+1);prefetch(i+2)  # keep the read-ahead worker busy while this page letters
                    with Image.open(path) as im:
                        im.load();size=im.size;fmt=im.format;multiframe=getattr(im,'n_frames',1)>1
                        if im.width*im.height>100_000_000:raise ValueError('图片超过一亿像素，无法安全载入。')
                        image=im.convert('RGB');alpha=im.getchannel('A') if 'A' in im.getbands() else None
                    cover=path==first_page and cover_candidate(image,self.vision)
                    if cover:result=image;regions=[];note='封面海报原样保留';self.summary['covers']+=1
                    elif multiframe:result=image;regions=[];note='多帧图片保留原文件，避免丢帧'
                    else:
                        # No whole-page second pass: region-level OCR and translation retries
                        # already ran, and what remains is almost always effect lettering.
                        # Repeating the page roughly doubled the time of most pages.
                        future=ahead.pop(str(path),None);parts=None
                        if future is not None:
                            prepared=future.result()
                            if prepared and prepared['digest']==digest:parts=prepared['parts']
                        if parts is None:parts=self.analyze(image)
                        else:self.log('识别已在上一页翻译时完成')
                        result,regions=self.finish(image,*parts)
                        note='未检测到可可靠处理的文字' if not regions else ''
                    self.check()
                    if engine.sha256(path)!=digest:raise ValueError('处理期间原图发生变化，未写入旧结果。')
                    if alpha is not None:result.putalpha(alpha)
                    assert result.size==size
                    translated=sum(r.get('status')=='已嵌字' for r in regions)
                    preserved=sum(r.get('status')!='已嵌字' for r in regions)
                    expected=sum(expected_preservation(r) for r in regions if r.get('status')!='已嵌字')
                    residual=sum(r.get('status')=='已嵌字' and r.get('post_render_verified') is False for r in regions)
                    unresolved=preserved-expected+residual
                    # Preserve original bytes when a page requires no replacement at all.
                    record={'source_sha256':digest,'settings':self.settings,'size':list(size),
                            'translated_regions':translated,'preserved_regions':preserved,'intentional_skips':expected,'unresolved_regions':unresolved,
                            'retained':bool(unresolved),'note':note,
                            'preservation_reasons':dict(collections.Counter(r.get('automatic_reason') or r.get('status') for r in regions if r.get('status')!='已嵌字'))}
                    working=checked_path(self.job,self.job/'working'/rel.parent/(rel.name+'.json'))
                    metadata={'source':str(path),'output':str(dst),'source_sha256':digest,'language':self.language,
                              'job':str(self.job),'key':key,'regions':regions,'note':note,'size':list(size)}
                    writer=(lambda dest:copy_atomic(path,dest)) if multiframe or not translated else (lambda dest:save_matching(result,dest))
                    record=commit_page(self,key,dst,writer,record,metadata)
                    self.summary['completed']+=1;self.summary['retained_pages']+=int(record['retained']);self.summary['preserved_regions']+=preserved;self.summary['intentional_skips']+=expected
                    label='封面原样保留' if cover else ('已输出（有未译文字）' if record['retained'] else '完成')
                    self.checkpoint();self.event('done',str(path),str(dst),label,str(working))
                except engine.Cancelled:raise
                except Exception as exc:
                    self.summary['failed']+=1;self.log('文件未完成：'+str(exc));self.event('error',str(path),str(exc))
            self.companions();self.check()
            if not self.summary['failed']:
                for path in files:
                    rel=Path(path).relative_to(self.source);saved=self.state['pages'][str(rel)]
                    assert engine.sha256(path)==saved['source_sha256'],'原图在任务结束前发生变化'
                    assert engine.sha256(self.output/rel)==saved['output_sha256'],'输出在任务结束前发生变化'
                all_pages=scan_images(self.source)
                fully_saved=all(str(p.relative_to(self.source)) in self.state['pages'] and (self.output/p.relative_to(self.source)).is_file() for p in all_pages)
                self.checkpoint('complete' if fully_saved else 'incomplete')
            else:self.checkpoint('incomplete')
        except engine.Cancelled:self.summary['cancelled']=True;self.checkpoint('cancelled')
        finally:
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
            self.translator.close()
        return self.summary

    def analyze_file(self,path):
        """Read-ahead worker: its own Vision instance, no translator, no file writes."""
        if not hasattr(self,'vision_ahead'):self.vision_ahead=engine.Vision()
        digest=engine.sha256(path)
        with Image.open(path) as im:
            im.load()
            if getattr(im,'n_frames',1)>1 or im.width*im.height>100_000_000:return None
            image=im.convert('RGB')
        return {'digest':digest,'parts':self.analyze(image,vision=self.vision_ahead,log=lambda _:None)}
