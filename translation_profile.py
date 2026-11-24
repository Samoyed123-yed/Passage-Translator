"""Editable local translation policy, loaded once per batch; no model training."""
import json,re
from pathlib import Path

PROFILE_PATH=Path(__file__).resolve().parent/'translation_style.json'

def load_profile(path=None):
    path=Path(path or PROFILE_PATH)
    if path.stat().st_size>128_000:raise ValueError('翻译风格配置过大。')
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema_version')!=1:raise ValueError('翻译风格配置版本不支持。')
    if not isinstance(data.get('instructions'),list) or not all(isinstance(s,str) for s in data['instructions']):
        raise ValueError('翻译风格 instructions 必须是文字列表。')
    if not isinstance(data.get('terms'),list):raise ValueError('翻译风格 terms 必须是词条列表。')
    for row in data['terms']:
        if not isinstance(row,dict) or not isinstance(row.get('term'),str) or not row['term'] or not isinstance(row.get('guidance'),str):
            raise ValueError('术语必须含 term 和 guidance 文字。')
        if not isinstance(row.get('aliases',[]),list) or not all(isinstance(s,str) and s for s in row.get('aliases',[])):
            raise ValueError('术语 aliases 必须是非空文字列表。')
    # Sakura glossary: one target per source; Sakura does not support "a/b" alternatives.
    for row in data.get('sakura_terms',[]):
        if not isinstance(row,dict) or not all(isinstance(row.get(k),str) and row[k] for k in ('src','dst')) or '/' in row['dst']:
            raise ValueError('sakura_terms 每项需要 src 与 dst 文字，且 dst 只能有一个译法。')
        if not isinstance(row.get('note',''),str):raise ValueError('sakura_terms 的 note 必须是文字。')
    for key in ('general_instructions','adult_instructions'):
        if not isinstance(data.get(key,[]),list) or not all(isinstance(s,str) for s in data.get(key,[])):
            raise ValueError(f'翻译风格 {key} 必须是文字列表。')
    return data

AUDIENCES=('auto','general','adult')

def style_instructions(profile,audience='auto'):
    """auto (the app default): everyday tone plus the adult rule, which only speaks to
    adult dialogue. general / adult remain for scripts and tests."""
    keys={'general':['general_instructions'],'adult':['adult_instructions']}.get(audience,['general_instructions','adult_instructions'])
    return '\n'.join([*profile['instructions'],*[s for k in keys for s in profile.get(k,[])]])

def for_audience(rows,audience):
    """日常 (general) skips rows marked adult; 成人 keeps everything."""
    return [row for row in rows if audience!='general' or not row.get('adult')]

def relevant_terms(profile,text,audience='auto'):
    from sakura import fold
    text=fold(text)
    result={}
    for row in for_audience(profile['terms'],audience):
        for alias in [row['term'],*row.get('aliases',[])]:
            pattern=re.escape(fold(alias))
            if re.search('[A-Za-z]',alias):pattern=r'(?<![A-Za-z])'+pattern+r'(?![A-Za-z])'
            if re.search(pattern,text,re.I):
                result[row['term']]=row['guidance'];break
    return result
