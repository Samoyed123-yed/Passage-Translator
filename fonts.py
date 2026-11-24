"""Lettering font families (switchable in the UI, stored in preferences.json).

Each family lists three weights: light-source, regular-source, heavy-source lettering.
The default is Resource Han Rounded (SIL OFL), closest to the heavy rounded gothic that
scanlation groups typically use for dialogue.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = Path('C:/Windows/Fonts')
FAMILIES = {
    'rounded': ('圆体', [ROOT / 'fonts/ResourceHanRoundedCN-Medium.ttf', ROOT / 'fonts/ResourceHanRoundedCN-Medium.ttf', ROOT / 'fonts/ResourceHanRoundedCN-Bold.ttf']),
    'noto': ('黑体', [ROOT / 'fonts/NotoSansSC-Regular.otf', ROOT / 'fonts/NotoSansSC-Bold.otf', ROOT / 'fonts/NotoSansSC-Black.otf']),
    'yahei': ('雅黑', [WIN / 'msyh.ttc', WIN / 'msyhbd.ttc', WIN / 'msyhbd.ttc']),
    'youyuan': ('幼圆', [WIN / 'SIMYOU.TTF', WIN / 'SIMYOU.TTF', WIN / 'SIMYOU.TTF']),
}
DEFAULT = 'rounded'
_cache = {'mtime': None, 'family': None}

def available():
    return [key for key, (_, paths) in FAMILIES.items() if all(Path(p).is_file() for p in paths)]

def current():
    """Family chosen in preferences.json, falling back to the first installed family."""
    prefs = ROOT / 'preferences.json'
    try:
        mtime = prefs.stat().st_mtime
        if mtime != _cache['mtime']:
            _cache['family'] = json.loads(prefs.read_text(encoding='utf-8')).get('lettering_font'); _cache['mtime'] = mtime
    except (OSError, ValueError):
        _cache['family'] = None
    family = _cache['family'] or DEFAULT
    ok = available()
    return family if family in ok else (DEFAULT if DEFAULT in ok else (ok[0] if ok else 'yahei'))

def pick(level=1, family=None):
    """Font file for a weight level: 0 light source, 1 regular, 2 heavy."""
    paths = FAMILIES[family or current()][1]
    return paths[max(0, min(2, level))]


VOICE = WIN / 'STHUPO.TTF'  # 华文琥珀: heavy rounded display face, used for pure cries in balloons

def voice():
    """Expressive face for balloons that only hold a cry (啊♥, 呜哦♥), like the group's."""
    return VOICE if VOICE.is_file() else None
