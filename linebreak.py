"""Word-aware line breaking for Chinese lettering (jieba segmentation, offline).

Scanlation typesetters break at word and phrase boundaries ("让哥哥多看看 / 你可爱的地方
/ 吧♥"), never inside a word ("姐 / 姐") and never start a line with a particle or closing
punctuation. Lines are balanced: the longest line is as short as possible.
"""
import re
import unicodedata
from functools import lru_cache

NO_START = set('，。！？、；：）》」』】…—!?.,～~♡♥❤ー·・')
OPENING = set('（《「『【“‘(')
PARTICLES = set('的了吗呢吧啊嘛呀哦着过们得地喔哟呗啦哈嗯么')
BREAK_AFTER = set('，。！？、；：…!?')

@lru_cache(maxsize=1)
def _jieba():
    try:
        import jieba
        jieba.setLogLevel(60)
        return jieba
    except ImportError:
        return None

@lru_cache(maxsize=2048)
def tokens(text):
    """Unbreakable units: words, with trailing punctuation/particles/hearts glued on."""
    text = re.sub(r'\s+', '', text)
    if not text: return ()
    jb = _jieba()
    words = list(jb.cut(text, HMM=True)) if jb else list(text)
    out = []
    for w in words:
        if not w: continue
        glue = out and (w[0] in NO_START or (len(w) == 1 and w in PARTICLES) or out[-1][-1] in OPENING)
        if glue: out[-1] += w
        else: out.append(w)
    spans = [(m.start(), m.end()) for w in PROTECTED for m in re.finditer(re.escape(w), text)]
    if spans:
        merged, pos = [], 0
        for t in out:
            start = pos; pos += len(t)
            if merged and any(a < start < b for a, b in spans): merged[-1] += t
            else: merged.append(t)
        out = merged
    return tuple(out)

PROTECTED = set()

def protect(words):
    """Character names that must never be split across lines (天咲 / 光)."""
    new = {w for w in words if len(w) >= 2} - PROTECTED
    if new: PROTECTED.update(new); tokens.cache_clear()

def names_in(source, translation):
    """Kanji names shared by the Japanese and the Chinese. A name whose Japanese continues
    in kana (天咲ひかり -> 天咲光) keeps the next Chinese character with it."""
    source = unicodedata.normalize('NFKC', source or ''); found = set()
    for m in re.finditer(r'[一-鿿]{2,4}', source):
        word = m.group(); after = source[m.end():m.end() + 1]
        for t in re.finditer(re.escape(word), translation or ''):
            nxt = translation[t.end():t.end() + 1]
            kana_name = re.fullmatch('[ぁ-ゖァ-ヺ]', after) and after not in 'のはがをにでともへやかね'
            found.add(word + nxt if kana_name and re.fullmatch('[一-鿿]', nxt) and nxt not in PARTICLES and nxt not in '是在和' else word)
    return found

def _units(text):
    return [len(t) for t in tokens(text)]

BIND_NEXT = set('被让很多把在和跟给对向想要会能不没也都还就才又再最太更真好')  # never end a line on these
PRONOUN_START = set('你我他她它这那咱')

def _break_bonus(left, right):
    """Typesetter's preference for a line break between two tokens."""
    bonus = 0.
    if left[-1] in BREAK_AFTER or left[-1] in '♥♡❤': bonus += 2.5
    if right[0] in PRONOUN_START: bonus += 1.2
    if len(left) == 1 and left in BIND_NEXT: bonus -= 3.0
    if left[-1] in PARTICLES and left[-1] != '的': bonus += .6
    if left[-1] == '的' and right[0] not in BREAK_AFTER: bonus -= 1.6  # 的 belongs to the noun after it
    return bonus

def partition(text, n):
    """Split text into exactly n lines at token boundaries. Cost = 3 x longest line +
    unevenness - break preferences, so a phrase boundary may win over a one-character
    shorter line (as a human typesetter does). None if impossible."""
    toks = tokens(text)
    if n <= 0 or n > len(toks): return None
    lens = [len(t) for t in toks]; avg = sum(lens) / n; m = len(toks)
    bonus = [_break_bonus(toks[j], toks[j + 1]) if j + 1 < m else 0. for j in range(m)]
    best_total, best_split = None, None
    for cap in sorted({sum(lens[a:b]) for a in range(m) for b in range(a + 1, m + 1)}):
        if cap < max(lens): continue
        @lru_cache(maxsize=None)
        def go(i, k):
            if k == 1:
                L = sum(lens[i:]); return (.15 * (L - avg) ** 2, ()) if L <= cap else (float('inf'), ())
            res = (float('inf'), ()); run = 0
            for j in range(i, m - k + 1):
                run += lens[j]
                if run > cap: break
                rest, splits = go(j + 1, k - 1)
                cost = .15 * (run - avg) ** 2 - bonus[j] + rest
                if cost < res[0]: res = (cost, (j + 1,) + splits)
            return res
        cost, splits = go(0, n)
        if cost == float('inf'): continue
        total = 3 * cap + cost
        if best_total is None or total < best_total: best_total, best_split = total, splits
        if best_total is not None and 3 * cap > best_total + 12: break
    if best_split is None: return None
    edges = (0,) + best_split + (m,)
    return [''.join(toks[a:b]) for a, b in zip(edges, edges[1:])]

def longest_unit(text):
    lens = _units(text); return max(lens) if lens else 0

def quality(lines):
    """Sum of break preferences between consecutive lines (higher reads more naturally)."""
    return sum(_break_bonus(a, b) for a, b in zip(lines, lines[1:]) if a and b)

def char_partition(text, n):
    """Fallback when a single word is longer than a line can hold."""
    text = re.sub(r'\s+', '', text); size = -(-len(text) // n); out = []
    while text:
        cut = min(size, len(text))
        while cut < len(text) and text[cut] in NO_START: cut += 1
        out.append(text[:cut]); text = text[cut:]
    return out
