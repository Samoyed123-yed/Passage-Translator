import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import engine
from text_rules import *
from typesetting import BubbleMap,fit_text

class RepairTests(unittest.TestCase):
    def test_punctuation_preserved_but_failed_short_reviewable(self):
        for text in ('……','・・・','．．．','︙','ーー','！？'):
            r={'source':text,'translation':'错误译文','status':'已翻译'}
            self.assertTrue(normalize_preserved(r)); self.assertFalse(show_in_editor(r)); self.assertEqual(review_count([r]),0)
        r={'source':'もしもし？','translation':'喂？','status':'已翻译'}
        self.assertFalse(normalize_preserved(r)); self.assertEqual(r['translation'],'喂？')
        r={'source':'誰','translation':'','status':'需检查译文'}
        self.assertFalse(normalize_preserved(r)); self.assertTrue(show_in_editor(r)); self.assertEqual(review_count([r]),1)

    def test_missing_long_translation_still_needs_review(self):
        r={'source':'今日は雨が降るから早く帰りましょう。','translation':'','status':'需检查译文'}
        self.assertFalse(normalize_preserved(r)); self.assertEqual(review_count([r]),1)

    def test_request_ids_do_not_use_page_ids(self):
        # Qwen JSON request path (used for en/ko, and ja when the Sakura model is absent).
        translator=engine.LocalTranslator(lambda _:None); translator.process=True; translator.url='http://127.0.0.1'; translator.backend='qwen'
        regions=[{'id':11,'source':'こんにちは。'},{'id':27,'source':'おやすみなさい。'}]
        class Response:
            def raise_for_status(self): pass
            def json(self): return {'choices':[{'message':{'content':json.dumps({'items':[{'id':0,'kind':'dialogue','zh':'你好。'},{'id':1,'kind':'dialogue','zh':'晚安。'}]})}}]}
        def post(url,**kwargs):
            body=json.loads(kwargs['json']['messages'][1]['content'])
            self.assertEqual([r['id'] for r in body['items']],[0,1]); return Response()
        with patch.object(translator.http,'post',post): translator.translate(regions,'ja','',[])
        self.assertEqual([r['translation'] for r in regions],['你好。','晚安。'])
        self.assertTrue(invalid_translation('えっ','这是一整段完全不属于原句的很长解释性文字。'))
        translator.http.close()

    def test_ellipsis_pixels_and_real_text(self):
        im=Image.new('RGB',(30,120),'white'); d=ImageDraw.Draw(im)
        for y in (12,32,52,72,92): d.ellipse((12,y,18,y+6),fill='black')
        self.assertTrue(image_is_ellipsis(im))
        im=Image.new('RGB',(180,50),'white'); ImageDraw.Draw(im).text((2,2),'こんにちは',font=ImageFont.truetype(str(engine.FONT),25),fill='black')
        self.assertFalse(image_is_ellipsis(im))

    def test_irregular_bubble_center_and_fit(self):
        im=Image.new('RGB',(900,900),'#888888'); d=ImageDraw.Draw(im)
        d.polygon([(80,90),(250,55),(420,100),(390,200),(430,280),(360,420),(190,395),(60,300),(100,240)],fill='white',outline='black',width=4)
        r={'box':[210,170,270,300]}; bubbles=BubbleMap(im,[r]); b=bubbles.find(r)
        self.assertIsNotNone(b)
        for vertical in (True,False):
            f=fit_text('我们应该在下雨之前赶快回家。',r['box'],vertical,engine.FONT,b,'雨が降る前に帰ろう。')
            self.assertIsNotNone(f)
            x,y,x2,y2=f['bounds']; cx,cy=b['center']
            self.assertLess(abs((x+x2)/2-cx),1); self.assertLess(abs((y+y2)/2-cy),1)
            mask=np.asarray(f['tile'])>0; bx,by,*_=b['box']
            self.assertTrue(b['safe'][y-by:y2-by,x-bx:x2-bx][mask].all())
        short=fit_text('好的。',r['box'],True,engine.FONT,b,'我会尽快赶到那里。')
        long=fit_text('我会尽快赶到那里和大家一起等候。',r['box'],True,engine.FONT,b,'我会尽快赶到那里。')
        self.assertGreater(short['size'],long['size'])

    def test_floating_bounds_and_page_margin_not_bubble(self):
        im=Image.new('RGB',(300,300),'white'); r={'box':[120,30,155,220]}
        self.assertIsNone(BubbleMap(im,[r]).find(r))
        f=fit_text('我们应该趁下雨之前一起回家。',r['box'],True,engine.FONT)
        self.assertIsNotNone(f)
        x,y,x2,y2=f['bounds']; self.assertTrue(x>=120 and x2<=155 and y>=30 and y2<=220)

    def test_thin_outline_does_not_join_adjacent_white_artwork(self):
        im=Image.new('RGB',(600,600),'#888888');d=ImageDraw.Draw(im)
        d.rectangle((100,100,220,400),fill='white',outline='black',width=1)
        d.line((100,240,220,240),fill='black',width=1)
        r={'box':[130,130,180,210]};b=BubbleMap(im,[r]).find(r)
        self.assertIsNotNone(b);self.assertLessEqual(b['box'][3],240)

    def test_skipped_region_survives_overlapping_cleaning(self):
        im=Image.new('RGB',(400,400),'#888888');d=ImageDraw.Draw(im)
        d.ellipse((50,50,350,350),fill='white',outline='black',width=2)
        for y in (116,128,140):d.ellipse((210,y,214,y+4),fill='black')
        mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle((120,100,280,260),fill=255)
        regions=[{'id':0,'source':'今日は晴れです','box':[120,100,280,260],'translation':'今天晴天','kind':'dialogue','direction_override':'horizontal'},
                 {'id':1,'source':'…','box':[200,110,235,150],'translation':'','status':'短字或符号，保留原文'}]
        class Plain:
            def inpaint(self,image,mask):return Image.new('RGB',image.size,'white')
        out=engine.render(im,mask,regions,Plain(),'ja')
        self.assertEqual(out.crop(regions[1]['box']).tobytes(),im.crop(regions[1]['box']).tobytes())
        self.assertEqual(regions[0]['status'],'已嵌字')

if __name__=='__main__': unittest.main()
