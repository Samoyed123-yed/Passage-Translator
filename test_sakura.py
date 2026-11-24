import unittest
from unittest.mock import patch
import engine, sakura
from translation_profile import load_profile

class Reply:
    def __init__(self,text):self.text=text
    def raise_for_status(self):pass
    def json(self):return {'choices':[{'message':{'content':self.text}}]}

class SakuraTests(unittest.TestCase):
    def translator(self,replies):
        t=engine.LocalTranslator(lambda _:None);t.process=object();t.backend='sakura';t.url='http://127.0.0.1:1'
        calls=[]
        def post(url,json,timeout):
            calls.append(json);return Reply(replies.pop(0))
        t.http.post=post
        return t,calls

    def test_glossary_matches_either_kana_script_and_user_entries_win(self):
        rows=sakura.glossary_rows(load_profile(),'ちんぽ = 肉棒\n美波 = 美波 # 男性','チンポしごけ美波')
        self.assertIn('ちんぽ->肉棒',rows);self.assertIn('しごけ->撸 #命令语气，用手撸',rows);self.assertIn('美波->美波 #男性',rows)
        self.assertFalse(any(r.startswith('ちんぽ->鸡巴') for r in rows))

    def test_swapped_lines_with_equal_count_are_not_aligned(self):
        lines=['お前のＳフォロワＩＮＳの','最近エロ自撮りの更新がなくて寂しがってると思ってな']
        self.assertFalse(sakura.aligned(lines,['我想你最近没有更新色情自拍，','你的Ｓ粉丝ＩＮＳ应该会很寂寞吧']))
        self.assertTrue(sakura.aligned(lines,['你的Ｓ粉丝ＩＮＳ的','我想你最近没更新色情自拍挺寂寞的']))

    def test_misaligned_page_falls_back_to_one_request_per_line(self):
        t,calls=self.translator(['只有一行','第一句','第二句'])
        regions=[{'id':0,'source':'ここ外なのに','kind':'dialogue'},{'id':1,'source':'人が通ってるのに','kind':'dialogue'}]
        with patch.object(engine,'normalize_preserved',return_value=False),patch.object(sakura,'available',return_value=True):
            t.translate(regions,'ja','',[])
        self.assertEqual(len(calls),3)
        self.assertEqual([r['translation'] for r in regions],['第一句','第二句'])
        self.assertEqual(calls[1]['messages'][0]['content'],sakura.SYSTEM)

    def test_parenthesized_or_short_katakana_output_is_sound_effect(self):
        t,_=self.translator(['（弹弹）\n这里可是外面啊．．．'])
        regions=[{'id':0,'source':'パンパッ','kind':'dialogue','decorative_candidate':True,'visual_evidence':{'class':2,'line_support':.1}},{'id':1,'source':'ここ外なのに．．．','kind':'dialogue'}]
        with patch.object(engine,'normalize_preserved',return_value=False),patch.object(sakura,'available',return_value=True):
            t.translate(regions,'ja','',[])
        self.assertEqual(regions[0]['kind'],'sfx');self.assertEqual(regions[0]['translation'],'')
        self.assertEqual(regions[1]['translation'],'这里可是外面啊……')

    def test_boxed_voice_is_never_dropped_as_sound_effect(self):
        self.assertFalse(sakura.looks_like_sfx({'source':'アッ','boxed_voice':True},'啊'))

    def test_review_pass_is_skipped_for_sakura_output(self):
        t,calls=self.translator([])
        r={'source':'いくっ','translation':'要去了','kind':'dialogue'}
        t.review_translations([r],'ja');self.assertEqual(calls,[]);self.assertEqual(r['translation'],'要去了')

    def test_other_languages_keep_qwen(self):
        t=engine.LocalTranslator(lambda _:None)
        with patch.object(sakura,'available',return_value=True):
            self.assertFalse(t.use_sakura('en'));self.assertTrue(t.use_sakura('ja'))

if __name__=='__main__':unittest.main()
