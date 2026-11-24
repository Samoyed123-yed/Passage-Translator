"""Conservative preservation rules shared by translation, rendering and proofreading."""
import re
import unicodedata
import cv2
import numpy as np

def image_is_ellipsis(crop):
    gray=np.asarray(crop.convert('L'))
    # Only isolated, similarly-sized round marks arranged on one axis qualify.
    if float((gray>210).mean())<.65: return False
    n,_,stats,centers=cv2.connectedComponentsWithStats((gray<145).astype(np.uint8),8)
    stats=stats[1:]; centers=centers[1:]
    keep=stats[:,4]>=3; stats=stats[keep]; centers=centers[keep]
    if not 3<=len(stats)<=18: return False
    widths=stats[:,2]; heights=stats[:,3]; areas=stats[:,4]
    if widths.max()>22 or heights.max()>22: return False
    if np.any((widths/heights<.55)|(widths/heights>1.8)|(areas/(widths*heights)<.45)): return False
    if areas.max()>areas.min()*3.5: return False
    spread=np.ptp(centers,axis=0); diameter=float(max(np.median(widths),np.median(heights)))
    return bool(spread.max()>diameter*3 and spread.min()<diameter*1.6)

def decorative_without_mask(crop,mask):
    # CTD rejects the stroke mask of colored handwritten effects; do not invent OCR prose.
    if float((np.asarray(mask)>64).mean())>.002: return False
    rgb=np.asarray(crop.convert('RGB')).astype(float)
    colored=(rgb.max(2)-rgb.min(2)>65)&(rgb.min(2)<180)
    ink=rgb.mean(2)<210
    return bool(colored.sum()>12 and colored.sum()/max(1,ink.sum())>.48)

QUIET_STATUSES={'已嵌字','跳过特效字','跳过不规则文字','短字或符号，保留原文',
                '跳过标点，保留原文','短句无法可靠翻译，保留原文','无需翻译，保留原文','自动保留原文'}

def lexical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text or '')
                   if unicodedata.category(c)[0] in 'LN' and c not in 'ー')

def punctuation_only(text):
    text=unicodedata.normalize('NFKC',text or '').strip()
    return bool(text) and not lexical(text)

def short_fragment(text):
    return len(lexical(text))<=8

def vocalization_only(text):
    """Whole utterances only: never remove a refusal or words around a cry."""
    s=unicodedata.normalize('NFKC',text or '').strip()
    if not s or len(s)>256:return False
    s=re.sub(r'[\s♡♥❤!！?？.,，。…・·~～ー—っッぁァぅゥぉォ]', '',s)
    if not s:return False
    if s in {'ううん','ウウン'}:return False  # A colloquial denial, not a moan.
    return bool(re.fullmatch(r'(?:ふい|フイ|ひい|ヒイ)+',s) or re.fullmatch(r'[あアうウおオんンはハふフ]+',s) or
                re.fullmatch(r'[啊呀哦噢喔嗯唔呜哈哼吼]+',s) or
                re.fullmatch(r'(?:하아|하악|하앗|허억|흐읏|후우|으윽|으읏|아앗|하읏|으응|아|하|흐)+',s) or
                re.fullmatch(r'(?:a+h+|h+a+|u+h+|o+h+|h+m+|m+m+|n+g+h+)+',s,re.I))

def preserve(r,reason):
    r.update(translation='',kind='preserve',status=reason,skip_review=True)

def mark_boxed_voice(image,regions,r):
    """Only recognized pure vocalizations fully inside a bordered balloon qualify."""
    r.pop('boxed_voice',None)
    r.pop('boxed_prose',None)
    text=r.get('source','')
    if r.get('visual_evidence',{}).get('class')==1:
        r['boxed_prose']=True
        if vocalization_only(text):r['boxed_voice']=True
        return bool(text)
    from typesetting import BubbleMap
    bubble=BubbleMap(image,regions).find(r)
    # A small explicit effect lexicon is safe only outside a confirmed balloon.
    # Do not treat unknown kana, failed OCR, or floating narration as sound effects.
    compact=re.sub(r'[\s!！?？…・ー〜～っッ]','',unicodedata.normalize('NFKC',text))
    if bubble is None and re.fullmatch(r'(?:がく|ガク|どき|ドキ|びく|ビク|ごく|ゴク|ぞく|ゾク){1,6}',compact):
        r.update(translation='',kind='sfx',status='跳过特效字',skip_review=True,confirmed_effect=True)
        r.pop('automatic_reason',None)
        return False
    if not bubble or not bubble.get('bordered'):return False
    x,y,X,Y=map(int,r['box']);bx,by,bX,bY=bubble['box']
    if x<bx or y<by or X>bX or Y>bY:return False
    if float(bubble['body'][y-by:Y-by,x-bx:X-bx].mean())<.95:return False
    if not vocalization_only(text):
        if re.search(r'[\u4e00-\u9fff]',text) and len(lexical(text))>=3:
            r['boxed_prose']=True
            r.pop('skip_review',None);r.pop('automatic_reason',None)
            if r.get('kind') in ('sfx','preserve'):r['kind']='dialogue'
            return True
        return False
    r['boxed_voice']=True
    r.pop('skip_review',None);r.pop('automatic_reason',None)
    if r.get('kind') in ('sfx','preserve'):r['kind']='dialogue'
    return True

def normalize_preserved(r,failed_short=True):
    text=r.get('source','')
    if r.get('confirmed_effect'):
        r.update(translation='',kind='sfx',status='跳过特效字',skip_review=True)
        r.pop('automatic_reason',None)
        return True
    if vocalization_only(text):
        if r.get('boxed_voice'):
            r.pop('skip_review',None)
            return False
        r.update(translation='',kind='sfx',status='跳过特效字',skip_review=True)
        r.pop('automatic_reason',None)
        return True
    if punctuation_only(text): preserve(r,'跳过标点，保留原文'); return True
    chars=lexical(text)
    if chars and chars.isdecimal(): preserve(r,'无需翻译，保留原文'); return True
    if sum(w in chars for w in ('無断','転載','転職','禁止','生成','学習'))>=2 and ('無断' in chars or '禁止' in chars):
        # Publisher footer (無断転載禁止・生成AI学習禁止), not part of the story.
        preserve(r,'无需翻译，保留原文'); return True
    x1,y1,x2,y2=r.get('box') or (0,0,0,0)
    if (r.get('ocr_reading_order')=='large-line-rescaled' and y2-y1>=140 and x2-x1>=(y2-y1)*1.6
            and 0<len(chars)<=8):
        # One short row of big outlined art lettering over a drawing (うるうるの / 黒目 /
        # 華奢な白い首筋): user rule 2026-11-24, these descriptors stay as drawn.
        preserve(r,'艺术字，保留原文'); return True
    if len(chars)>=3 and re.fullmatch('[一-鿿]',chars[0]) and chars.count(chars[0])>=len(chars)*.8:
        # One kanji repeated as a visual gag (男男男男…) reads the same in Chinese;
        # redrawing it only risks clipping the balloon.
        preserve(r,'无需翻译，保留原文'); return True
    if r.get('pen_note') and 0<len(chars)<=16 and (r.get('ocr_confidence') or 0)<.92:
        # A short pen/pencil aside drawn beside the art (user rule: decorative handwritten
        # notes stay as drawn). Printed gray narration reads at >=0.99 and is kept.
        preserve(r,'无需翻译，保留原文'); return True
    if r.get('source_language')=='ja' and len(re.findall('[A-Za-z]',chars))>=4 and len(re.findall('[A-Za-z0-9]',chars))>=len(chars)*.6:
        # Latin lettering in a Japanese book is a logo, signature or watermark
        # (NO REPOST / artist name); manga-ocr turns it into kanji noise.
        preserve(r,'无需翻译，保留原文'); return True
    if chars and lexical(r.get('translation',''))==chars and re.fullmatch('[一-鿿]+',chars):
        # A kanji name plate (依野＿真夜) reads the same in Chinese: re-lettering it
        # only adds cleaning marks.
        preserve(r,'无需翻译，保留原文'); return True
    # Length alone cannot distinguish a failed sentence from a decorative mark.
    # Keep failures reviewable so the batch recovery path retries them.
    if r.get('status')=='短字或符号，保留原文':
        preserve(r,'短字或符号，保留原文'); return True
    return r.get('kind')=='preserve'

def needs_review(r):
    return r.get('status','') not in QUIET_STATUSES

def review_count(regions):
    return sum(needs_review(r) for r in regions) if regions else 1

def show_in_editor(r):
    return not (r.get('skip_review') or punctuation_only(r.get('source','')) or
                r.get('status') in {'短字或符号，保留原文','短句无法可靠翻译，保留原文','无需翻译，保留原文'})

def normalize_romanized_names(source,zh):
    """Preserve unknown names consistently; never guess their kanji spelling."""
    if not isinstance(zh,str):return zh
    for name,suffix in re.findall(r'\b([A-Z]{2,}?)[- ]?(CHAN|KUN)\b',source.upper()):
        token=name+suffix
        zh=re.sub(r'\b'+re.escape(token)+r'\b',name.title()+('酱' if suffix=='CHAN' else '君'),zh,flags=re.I)
    zh=re.sub(r'(我叫|我的名字是)\s*([A-Z]{3,})',lambda m:m[1]+m[2].title(),zh)
    return zh

def source_grammar_hint(source):
    # Restrict checks to clear sentence-final colloquial constructions; questions,
    # standalone ねえ (hey), and positive requests such as 言ってね are excluded.
    if re.search(r'(?:させんな|すんな|するな)(?:よ|よな)?[。！!]*$',source):
        return '这是禁止句；んな在这里来自るな，表示不要让某事发生。中文必须保留别/不要之类的禁止，不能译成肯定请求。'
    if re.search(r'[てで](?:い)?ね[えぇ]+ー*(?:よ|ぞ|んだ)?[。！!]*$',source):
        return '句末てねえ/でねえ是ていない/でいない的口语缩略，表示事实否定。中文应表达没做过或没有在做，不是命令对方别做。'
    if re.search(r'前に',source):
        return '完整保留“做某事之前”的时间关系和事件。用自然的“……之前”表达；原文未表达紧迫时，不额外加入“快、赶紧、即将”。不能只剩一个催促动作。'
    return ''

def sakura_input(source):
    """Sakura reads a sentence-final ……ではある (affirmative: "it is, admittedly") as
    ではない and negates the line; give it the plain affirmative instead."""
    return re.sub(r'ではある(?=[\s…．.。・：:！!？?]*$)','だ',re.sub(r'\s+','',source))

def meaning_problem(source,zh):
    if not isinstance(zh,str):return ''
    hint=source_grammar_hint(source)
    if hint.startswith('这是禁止句') and not re.search(r'别|不要|不许|不准|禁止|不得|甭|莫',zh):return hint
    if hint.startswith('句末') and (not re.search(r'没|未|不|哪|谁说|才怪|个屁',zh) or re.match(r'^(?:你)?(?:别|不要|不许)',zh)):
        return hint
    return ''

def translation_input(source):
    """Restore only unambiguous sentence-final contractions, keeping source intact."""
    hint=source_grammar_hint(source)
    if hint.startswith('句末'):
        return re.sub(r'([てで])(?:い)?ね[えぇ]+ー*(?=(?:よ|ぞ|んだ)?[。！!]*$)',r'\1いない',source)
    if hint.startswith('这是禁止句'):
        return re.sub(r'(させ|す)んな(?=(?:よ|よな)?[。！!]*$)',lambda m:'させるな' if m[1]=='させ' else 'するな',source)
    return source

def invalid_translation(source,zh):
    if not isinstance(zh,str) or not re.search(r'[\u4e00-\u9fff]',zh): return True
    if (vocalization_only(zh) and not vocalization_only(source) and
            lexical(source).lower() not in {'yes','yep','yeah','yup','はい','うん','네','예'} and
            # へえ… / ほう / あら: a short kana interjection may become 哦…… / 哎呀.
            not re.fullmatch('[ぁ-ゖァ-ヺ]{1,3}',lexical(source))):return True
    if meaning_problem(source,zh):return True
    # \u30fb (U+30FB) is a bullet / name separator Chinese uses too, not kana.
    if re.search(r'[\u3040-\u30fa\u30fc-\u30ff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]',zh): return True
    # Full-width Latin (\uff34\uff2f\uff2b\uff2f\uff2e\uff2f\uff2b\uff2f) in the source still counts as Latin.
    if not re.search('[a-zA-Z]',unicodedata.normalize('NFKC',source)) and re.search('[a-zA-Z]{3,}',zh): return True
    if any(word.islower() for word in re.findall(r'[A-Za-z]{3,}',zh)):return True
    # Uppercase comic dialogue is still ordinary language, not automatically a name.
    ordinary={'TRAP','CREAMPIE','LOSER','FRIEND','BROTHER','SISTER','LITTLE','GIRL','BOY','WHAT','WHY','YES','YEP','HELLO','PLEASE','SORRY','THANKS','GONNA','WANNA'}
    if ordinary.intersection(re.findall(r'[A-Za-z]{3,}',zh.upper())):return True
    # A two-character interjection must not acquire a paragraph of unrelated context.
    if len(lexical(zh))>max(8,len(lexical(source))*2.4+4): return True
    return False
