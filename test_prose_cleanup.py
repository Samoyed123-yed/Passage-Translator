import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import engine
from text_rules import vocalization_only,normalize_preserved,invalid_translation
from text_style import prose_erase_mask
from typesetting import fit_adaptive

class ProseCleanupTests(unittest.TestCase):
    def test_unreliable_inpainting_falls_back_inside_owned_mask(self):
        im=Image.fromarray(np.random.default_rng(7).integers(40,180,(180,220,3),dtype=np.uint8))
        r={'id':0,'box':[40,35,175,145],'source':'これは背景の説明です','translation':'背景说明','kind':'narration','direction_override':'horizontal','source_font_size':24}
        mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle((70,65,130,110),fill=255)
        class BadRepair:
            def inpaint(self,image,mask):return image.copy()
        out=engine.render(im,mask,[r],BadRepair(),'ja',automatic=True)
        self.assertEqual(r['cleaning_mode'],'compact-white-fallback')
        self.assertEqual(out.getpixel((42,37)),im.getpixel((42,37)))
        self.assertEqual(out.getpixel((72,67)),(255,255,255))
    def test_only_pure_vocals_are_preserved(self):
        for source in ('ああっ…♡','んっ','アアア！','하아 하아','AHH!','嗯嗯，啊啊'):
            r={'source':source,'translation':'啊','kind':'dialogue'}
            self.assertTrue(normalize_preserved(r),source)
            self.assertEqual(r['translation'],'');self.assertEqual(r['kind'],'sfx')
        for source in ('あっ、待って！','いや、やめて。','ううん','不要啊！','아니야','OH, WAIT!','あの日は雨だった。'):
            self.assertFalse(vocalization_only(source),source)
        self.assertTrue(invalid_translation('今日は一緒に帰りましょう。','啊啊'))
        self.assertFalse(invalid_translation('YEP','嗯'))

    def test_source_block_is_erased_even_with_missing_bottom_row_mask(self):
        im=Image.new('RGB',(260,240),'#888888');d=ImageDraw.Draw(im)
        for y in (70,115,160):d.rectangle((60,y,190,y+12),fill='black')
        mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle((60,70,190,82),fill=255)
        r={'id':0,'box':[55,65,195,180],'source':'これは三行に分かれた普通の文章です。','translation':'知道了',
           'kind':'dialogue','direction_override':'horizontal'}
        class Repair:
            def inpaint(self,image,mask):return Image.new('RGB',image.size,'#888888')
        out=engine.render(im,mask,[r],Repair(),'ja',automatic=True)
        self.assertIn(r['cleaning_mode'],('local-planar-fill','compact-white-fallback'))
        self.assertEqual(out.getpixel((65,165)),(136,136,136))
        self.assertEqual(out.getpixel((65,75)),(136,136,136))
        self.assertEqual(out.getpixel((10,10)),im.getpixel((10,10)))

    def test_missing_mask_does_not_paint_over_unverified_artwork(self):
        im=Image.new('RGB',(360,260),'#898989');d=ImageDraw.Draw(im)
        d.rectangle((70,70,245,200),outline='black',width=3)
        d.line((200,60,230,210),fill='red',width=8)
        prose={'id':0,'box':[60,60,260,210],'source':'これは背景に書かれた説明です。','translation':'这是背景上的说明。',
               'kind':'narration','direction_override':'horizontal'}
        effect={'id':1,'box':[195,50,240,220],'source':'ああっ','translation':'啊','kind':'dialogue'}
        class Repair:
            def inpaint(self,image,mask):return Image.new('RGB',image.size,'#898989')
        out=engine.render(im,Image.new('L',im.size),[prose,effect],Repair(),'ja',automatic=True)
        self.assertEqual(prose['status'],'清字掩膜缺失，保留原文');self.assertEqual(effect['kind'],'sfx')
        self.assertEqual(out.crop(effect['box']).tobytes(),im.crop(effect['box']).tobytes())

    def test_cleaning_does_not_expand_across_neighbor(self):
        r={'box':[20,20,90,90]};other={'box':[80,20,120,90]}
        keep=Image.new('L',(160,120));mask=prose_erase_mask(keep.size,(15,15,95,95),r,[r,other],keep)
        self.assertEqual(mask.getpixel((70,35)),0)
        self.assertEqual(mask.getpixel((20,35)),255)

    def test_sparse_short_translation_grows_inside_both_reading_directions(self):
        b={'box':[0,0,300,300],'center':(150,150),'safe':np.ones((300,300),np.uint8),'bordered':True}
        for vertical in (True,False):
            f=fit_adaptive('知道了',[110,100,190,200],vertical,engine.FONT,b,'長い原文の三行の説明')
            if vertical:
                # User rule (2026-11-15): vertical Japanese->Chinese keeps the source glyph
                # size and leaves space; it is not grown to fill the balloon.
                self.assertTrue(18<=f['size']<=34,f['size'])
            else:
                self.assertGreater(f['occupancy'],.30)
            self.assertEqual(f['vertical'],vertical)
            x,y,X,Y=f['bounds'];self.assertTrue(b['safe'][y:Y,x:X][np.asarray(f['tile'])>0].all())

    def test_tilted_japanese_prose_reaches_ocr(self):
        im=Image.new('RGB',(160,160),'white');mask=Image.new('L',im.size)
        ImageDraw.Draw(mask).polygon([(25,80),(85,20),(130,65),(70,125)],fill=255)
        class OCR:
            def recognize(self,*args):return '今日は雨です。'
        v=engine.Vision();r={'box':[0,0,160,160]}
        with patch.object(v,'ocr',return_value=OCR()):self.assertTrue(v.recognize(im,mask,r,'ja'))
        self.assertEqual(r['status'],'待翻译')

if __name__=='__main__':unittest.main()
