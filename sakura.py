"""Sakura-GalTransl (Japanese -> Simplified Chinese) request format and output checks.

Sakura is a fine-tuned translation model: it expects its own fixed system prompt and a
plain "src->dst #note" glossary, not Passage's long rule list. Lines go in and come out
one-for-one; alignment is verified before any line is accepted.
Model card: https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7 (CC-BY-NC-SA 4.0).
"""
import re
import unicodedata
from pathlib import Path

MODEL = Path(__file__).resolve().parent / 'models/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf'
NAME = 'Sakura-GalTransl-7B-v3.7'
SYSTEM = ('你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，'
          '注意不要混淆使役态和被动态的主语和宾语，不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。')
HISTORY_LINES = 8

def available():
    return MODEL.is_file()

def fold(text):
    """NFKC + katakana->hiragana, so glossary entries match either script."""
    text = unicodedata.normalize('NFKC', text or '')
    return ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)

def glossary_rows(profile, user_glossary, text, audience='auto'):
    """Only entries whose source occurs in this request; user entries win.
    日常 (general) books skip adult-only entries such as オカズ -> 撸管素材."""
    folded = fold(text); rows = {}
    for row in profile.get('sakura_terms', []):
        if audience == 'general' and row.get('adult'): continue
        if fold(row['src']) in folded:
            rows[row['src']] = row['src'] + '->' + row['dst'] + (' #' + row['note'] if row.get('note') else '')
    for line in (user_glossary or '').splitlines():
        match = re.match(r'\s*(.+?)\s*(?:->|=|＝)\s*([^#]+?)\s*(?:#\s*(.*))?$', line)
        if match and fold(match[1]) in folded:
            rows[match[1]] = match[1] + '->' + match[2] + (' #' + match[3] if match[3] else '')
    return list(rows.values())

def user_prompt(lines, history, rows):
    text = ''
    if history: text += '历史翻译：' + '\n'.join(history[-HISTORY_LINES:]) + '\n'
    text += '参考以下术语表（可为空，格式为src->dst #备注）：\n' + ''.join(r + '\n' for r in rows)
    return text + '\n根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n' + '\n'.join(lines)

def request_body(lines, history, rows, temperature=0.3):
    return {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_prompt(lines, history, rows)}],
            'temperature': temperature, 'top_p': 0.8, 'frequency_penalty': 0.1, 'max_tokens': 1200}

def split_output(content):
    content = re.sub(r'<think>.*?</think>', '', content or '', flags=re.S)
    return [line.strip() for line in content.split('\n') if line.strip()]

def _anchors(text):
    text = unicodedata.normalize('NFKC', text or '').upper()
    return {c for c in text if '一' <= c <= '鿿' or (c.isascii() and c.isalnum())}

def aligned(lines, out):
    """Same count AND no line shares clearly more kanji/Latin anchors with a neighbor's output.

    A page-level request once returned the right number of lines with two sentences
    swapped; counting lines alone cannot detect that.
    """
    if len(lines) != len(out): return False
    for i, source in enumerate(lines):
        own = len(_anchors(source) & _anchors(out[i]))
        for j in (i - 1, i + 1):
            if 0 <= j < len(out) and len(_anchors(source) & _anchors(out[j])) >= own + 2: return False
    return True

# Sakura sometimes keeps a katakana loanword as lowercase English ("激烈play"); the
# validator then rejects an otherwise correct line and the balloon stays Japanese.
LOANWORDS = {'play': '玩法', 'sex': '做爱', 'cosplay': '角色扮演', 'ok': '好', 'h': '色色'}

def clean(zh, source=''):
    zh = (zh or '').strip().strip('「」"')
    # A list bullet (・ ●) usually survives on the page because its dot is too small
    # for the text mask; copying it into the translation would print it twice.
    zh = re.sub(r'^[・●•·]+\s*', '', zh)
    # A vertical ellipsis at the end of a column is often read as a colon (ある：).
    if re.search('[：:]$', unicodedata.normalize('NFKC', source or '').strip()): zh = re.sub(r'[：:]$', '……', zh)
    zh = re.sub(r'[．.・。…](?:\s*[．.・。…])+', '……', zh)  # also spaced: 被看光了.  .  .
    # A sweat mark drawn like し after ？ is copied over as a stray kana (那是……？シ).
    zh = re.sub(r'(?<=[？！?!…。」])[ぁ-ゖァ-ヺ]{1,2}$', '', zh)
    # A lone source dot between words is a pause, not a decimal point or a middle dot.
    zh = re.sub(r'(?<=[一-鿿])[．.・](?=[一-鿿])', '……', zh)
    if not re.search('[A-Za-z]', source or ''):
        zh = re.sub(r'(?<![A-Za-z])([A-Za-z]+)(?![A-Za-z])',
                    lambda m: LOANWORDS.get(m[1].lower(), m[1]), zh)
    return zh

def looks_like_sfx(region, zh):
    """Interim text-only guard until a visual sound-effect gate exists (see HANDOFF).

    Sakura marks onomatopoeia it renders with full parentheses, and short pure-katakana
    marks outside a confirmed balloon are almost always sound effects.
    """
    if region.get('boxed_voice') or region.get('boxed_prose'): return False
    # Text spelling or parentheses alone cannot distinguish a character name or
    # parenthetical narration from an effect. The visual gate owns that decision.
    from visual_evidence import effect_gate
    if not effect_gate(dict(region)):return False
    if re.fullmatch(r'[（(][^（）()]*[）)]', zh.strip()): return True
    lexical = re.sub(r'[\W_]', '', unicodedata.normalize('NFKC', region.get('source', '')))
    return bool(lexical) and len(lexical) <= 6 and bool(re.fullmatch(r'[ァ-ヶー]+', lexical))


def emphasised_source(region):
    """Japanese set in visibly larger type (>= 1.3x the other columns), or None."""
    cols = region.get('ocr_columns') or []
    if len(cols) < 2: return None
    widths = sorted(c['width'] for c in cols); base = widths[len(widths) // 2] if len(widths) > 2 else widths[0]
    # Furigana beside a name widens its column without enlarging the characters; larger
    # type also spaces its characters further apart along the column.
    def pitch(c):
        n = len(re.sub(r'[\W_]', '', c['text'])); return c.get('length', 0) / n if n else 0
    pitches = sorted(pitch(c) for c in cols if pitch(c)); base_pitch = pitches[len(pitches) // 2] if pitches else 0
    big = [c for c in cols if c['width'] >= base * 1.3 and (not base_pitch or not pitch(c) or pitch(c) >= base_pitch * 1.15)]
    if not big or len(big) == len(cols): return None
    text = ''.join(c['text'] for c in big)
    return text if len(re.sub(r'[\W_]', '', text)) >= 2 else None

def locate(emph_zh, translation):
    """Span (start, end) of the emphasised words inside the whitespace-free translation."""
    full = re.sub(r'\s+', '', translation or ''); core = re.sub(r'[\W_♥♡❤]', '', emph_zh or '')
    if len(core) < 2 or not full: return None
    best = (0, 0, 0)
    for i in range(len(full)):
        for j in range(len(core)):
            n = 0
            while i + n < len(full) and j + n < len(core) and full[i + n] == core[j + n]: n += 1
            if n > best[0]: best = (n, i, i + n)
    n, a, b = best
    if n < max(2, int(len(core) * .6)): return None
    while b < len(full) and full[b] in '了的啊呀哦呢吧♥♡❤！？…。，': b += 1
    return a, b
