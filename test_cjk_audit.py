import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import engine
from text_rules import invalid_translation,normalize_preserved,review_count,translation_input
from typesetting import _chunks,fit_adaptive
from automatic import AutoBatch,unresolved_count

class CJKAudit(unittest.TestCase):
    def test_source_review_uses_local_ids_and_missing_result_stays_unresolved(self):
        import json
        t=engine.LocalTranslator(lambda _:None);t.process=True;t.url='http://127.0.0.1'
        t.profile={'instructions':['保留我指定的口语风格'], 'terms':[]}
        rows=[{'id':9,'source':'その夜、街は静かだった。','translation':'城市静得令人不安。','kind':'narration'},
              {'id':21,'source':'行くな','translation':'别去。','kind':'dialogue'}]
        class Response:
            def raise_for_status(self):pass
            def json(self):return {'choices':[{'message':{'content':json.dumps({'items':[{'id':0,'zh':'那天晚上城市很安静。'}]})}}]}
        with patch.object(t.http,'post',return_value=Response()) as post:
            t.review_translations(rows,'ja','本地术语')
            payload=json.loads(post.call_args.kwargs['json']['messages'][1]['content'])
            self.assertEqual([r['id'] for r in payload['items']],[0,1])
            self.assertEqual(payload['user_glossary'],'本地术语')
            self.assertIn('保留我指定的口语风格',post.call_args.kwargs['json']['messages'][0]['content'])
        self.assertEqual(rows[0]['translation'],'那天晚上城市很安静。')
        self.assertEqual(rows[0]['semantic_review'],'revised')
        self.assertEqual(rows[1]['translation'],'');self.assertEqual(rows[1]['semantic_review'],'unresolved')
        t.http.close()

    def test_hangul_jamo_residue(self):
        for text in ('你好ㅋㅋ','好吧ㅠㅠ','你好ᄀ'):
            self.assertTrue(invalid_translation('안녕',text))

    def test_failed_short_dialogue_is_retried_and_reported(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);source=root/'input';source.mkdir()
        batch=AutoBatch(source,root/'output','en',log=lambda _:None)
        r={'source':"DON'T GO",'translation':'','kind':'uncertain','status':'需检查译文','source_language':'en'}
        self.assertFalse(normalize_preserved(r));self.assertEqual(review_count([r]),1)
        with patch.object(batch.translator,'translate'),patch.object(batch.translator,'retry_translation',return_value=False) as retry:
            batch.translate_group([r]);retry.assert_called_once()
        self.assertEqual(unresolved_count([r]),1)
        batch.translator.close()

    def test_one_character_japanese_dialogue_reaches_translation(self):
        class OCR:
            def recognize(self,*args):return '誰'
        image=Image.new('RGB',(80,80),'white'); mask=Image.new('L',image.size)
        ImageDraw.Draw(mask).rectangle((15,15,60,60),fill=255)
        vision=engine.Vision();r={'box':[0,0,80,80]}
        with patch.object(vision,'ocr',return_value=OCR()):
            self.assertTrue(vision.recognize(image,mask,r,'ja'))
        self.assertEqual(r['status'],'待翻译')

    def test_layout_brackets_and_contour_both_directions(self):
        self.assertEqual(''.join(_chunks('这是（测试）内容',3)),'这是（测试）内容')
        self.assertTrue(all(not s.endswith(('（','「','“')) for s in _chunks('这是（测试）内容',3)))
        for vertical in (False,True):
            mask=Image.new('L',(240,240));ImageDraw.Draw(mask).ellipse((12,12,228,228),fill=1)
            bubble={'box':[0,0,240,240],'center':(120,120),'safe':np.asarray(mask)}
            fit=fit_adaptive('别去，我还没说完。',[65,65,175,175],vertical,engine.FONT,bubble,'原文句子',override='vertical' if vertical else 'horizontal')
            self.assertIsNotNone(fit);self.assertGreaterEqual(fit['size'],12)
            x,y,X,Y=fit['bounds'];self.assertTrue(bubble['safe'][y:Y,x:X][np.asarray(fit['tile'])>0].all())

if __name__=='__main__':unittest.main()
