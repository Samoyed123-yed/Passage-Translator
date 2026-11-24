import unittest
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from paragraphs import english_strips,english_columns,heart_shape
from typesetting import BubbleMap
from text_rules import normalize_preserved,invalid_translation
from automatic import AutoBatch,expected_preservation

FIXTURES=Path(__file__).parent/'tests/fixtures'

class EnglishRecovery(unittest.TestCase):
    def test_paragraph_retry_keeps_old_line_when_local_detection_misses_it(self):
        from english_coverage import EnglishCoverage
        reader=object.__new__(EnglishCoverage)
        def line(text,y,score=.8):return {'text':text,'box':[40,y,140,y+20],'glyph':20,'score':score,'poly':[[40,y],[140,y],[140,y+20],[40,y+20]]}
        originals=[line('FIRST',100),line('SECOND',135)]
        def local(text,y):return {'text':text,'box':[4,y,104,y+20],'glyph':20,'score':.9,'poly':[[4,y],[104,y],[104,y+20],[4,y+20]]}
        reader.lines=lambda image,**kwargs:[local('BEFORE',0),local('SECOND',57)]
        reader.alternate_line=lambda image,poly,text,score:(text,score)
        result=reader.refine_group(Image.new('RGB',(250,250),'white'),originals)
        self.assertEqual({l['text'] for l in result},{'FIRST','SECOND','BEFORE'})

    def test_name_case_normalization_does_not_hide_ordinary_english(self):
        from text_rules import normalize_romanized_names
        self.assertEqual(normalize_romanized_names('AKIRACHAN!', 'AKIRACHAN！'),'Akira酱！')
        self.assertEqual(normalize_romanized_names("I'M KENJI.",'我叫KENJI。'),'我叫Kenji。')
        self.assertEqual(normalize_romanized_names('PLUS, I AGREE.','PLUS，我同意。'),'PLUS，我同意。')

    def test_short_word_with_ocr_zero_is_recovered(self):
        from english_coverage import EnglishCoverage
        from types import SimpleNamespace
        reader=object.__new__(EnglishCoverage)
        reader.reader=lambda im:SimpleNamespace(boxes=[[[0,0],[42,0],[42,25],[0,25]]],txts=['D0'],scores=[.8])
        self.assertEqual(reader.lines(Image.new('RGB',(100,100),'white'))[0]['text'],'DO')

    def test_uppercase_word_left_in_translation_is_not_a_name(self):
        self.assertTrue(invalid_translation('IS THAT YOUR FRIEND?', '那是你的 FRIEND 吗？'))
        self.assertFalse(invalid_translation('IS THAT TOMOKI?', '那是 Tomoki 吗？'))

    def test_contraction_survives_full_page_filter(self):
        from english_coverage import EnglishCoverage
        from types import SimpleNamespace
        reader=object.__new__(EnglishCoverage)
        reader.reader=lambda im:SimpleNamespace(boxes=[[[0,0],[42,0],[42,25],[0,25]]],txts=["I'M"],scores=[.98])
        self.assertEqual(reader.lines(Image.new('RGB',(100,100),'white'))[0]['text'],"I'M")

    def test_residual_check_distinguishes_name_from_untranslated_word(self):
        from english_coverage import EnglishCoverage
        reader=object.__new__(EnglishCoverage)
        reader.lines=lambda im:[{'text':'Tomoki','box':[10,10,60,30],'score':.98},{'text':'FRIEND','box':[10,35,60,50],'score':.98}]
        r={'id':0,'box':[0,0,100,100],'source':'TOMOKI IS MY FRIEND','translation':'Tomoki 是我的 FRIEND','status':'已嵌字'}
        hits=reader.residuals(Image.new('RGB',(100,100)),[r])
        self.assertEqual([h['words'] for h in hits],[['FRIEND']])

    def test_heart_filter_does_not_remove_letter_v(self):
        import numpy as np
        heart=Image.new('L',(60,60));d=ImageDraw.Draw(heart)
        d.ellipse((5,5,30,31),fill=255);d.ellipse((29,5,54,31),fill=255);d.polygon([(5,20),(54,20),(30,54)],fill=255)
        self.assertTrue(heart_shape(np.asarray(heart)>0))
        letter=Image.new('L',(60,60));ImageDraw.Draw(letter).line([(6,5),(30,54),(54,5)],fill=255,width=5)
        self.assertFalse(heart_shape(np.asarray(letter)>0))

    def test_layout_keeps_romanized_name_and_closing_punctuation_together(self):
        from typesetting import _chunks
        lines=_chunks('我就能和Tomoki、他弟弟一起玩了！！',5)
        self.assertTrue(any('Tomoki' in line for line in lines))
        self.assertFalse(any(line and line[0] in '，。！？、!?' for line in lines))

    def test_black_outlined_text_remains_visible_on_flat_black_background(self):
        import engine,numpy as np
        im=Image.new('RGB',(400,220),'black');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',48)
        ImageDraw.Draw(im).text((100,75),'HELLO',font=font,fill='black',stroke_width=5,stroke_fill='white')
        box=[90,70,280,135];mask=Image.new('L',im.size);ImageDraw.Draw(mask).rectangle(box,fill=255)
        r={'id':0,'box':box,'source':'HELLO','translation':'你好','kind':'dialogue','status':'已翻译','source_language':'en','direction_override':'horizontal','source_font_size':48}
        class Vision:
            def inpaint(self,image,mask):return Image.new('RGB',image.size,'black')
        result=engine.render(im,mask,[r],Vision(),'en',automatic=True)
        pixels=np.asarray(result.crop(r['layout']['bounds']).convert('L'))
        self.assertEqual(r['layout']['color'],[0,0,0])
        self.assertGreater(int((pixels>200).sum()),300)

    @unittest.skipUnless((FIXTURES/"_004.webp-5.png").is_file(), "Private comic fixtures are not distributed")
    def test_real_tight_italic_outline_and_irregular_balloon_lines(self):
        for name,count in (('_002.webp-5',6),('_004.webp-0',8),('_004.webp-5',6)):
            with self.subTest(name=name):
                crop=Image.open(FIXTURES/(name+'.png'));mask=Image.open(FIXTURES/(name+'-mask.png'))
                strips,details,_=english_strips(crop,mask)
                self.assertEqual(len(strips),count)
                self.assertTrue(all(s.height>=15 and s.width>s.height*.6 for s in strips))
    def test_empty_ocr_is_not_a_skippable_short_sentence(self):
        r={'source':'','translation':'','status':'识别不确定，保留原文'}
        self.assertFalse(normalize_preserved(r))
    def test_english_short_dialogue_gets_a_retry(self):
        r={'source':'YEP.','source_language':'en','translation':'','status':'需检查译文'}
        self.assertFalse(normalize_preserved(r))
    def test_long_dialogue_misclassified_as_sfx_is_retried(self):
        class Translator:
            def translate(self,group,*args):
                for r in group:r.update(kind='sfx',translation='',status='跳过特效字')
            def retry_translation(self,r,*args):r.update(kind='dialogue',translation='那我去打个招呼！',status='已翻译');return True
        batch=object.__new__(AutoBatch);batch.translator=Translator();batch.language='en';batch.glossary='';batch.cancel=lambda:False
        r={'source':"I'LL GO GREET HIM, THEN!!",'translation':'','kind':'dialogue'}
        batch.translate_group([r]);self.assertEqual(r['translation'],'那我去打个招呼！')
    def test_expected_skip_is_separate_from_processing_failure(self):
        self.assertTrue(expected_preservation({'kind':'sfx','status':'跳过特效字'}))
        self.assertFalse(expected_preservation({'kind':'preserve','status':'自动保留原文','automatic_reason':'识别失败'}))
    def test_staggered_columns_are_not_interleaved(self):
        im=Image.new('RGB',(340,280),'white');mask=Image.new('L',im.size);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        for x,y,lines in ((4,65,['THIS IS','THE FIRST','TIME I','VISITED','THE CITY.']),(180,4,['I WAS','HERE','EARLIER.','IT LOOKS','DIFFERENT','TODAY.'])):
            for i,line in enumerate(lines):
                ImageDraw.Draw(im).text((x,y+i*32),line,font=font,fill='black')
                ImageDraw.Draw(mask).text((x,y+i*32),line,font=font,fill=255)
        columns=english_columns(im,mask);self.assertEqual(len(columns),2)
        self.assertLess(columns[0][2],columns[1][0]);self.assertGreater(columns[0][1],columns[1][1])
    @unittest.skipUnless((FIXTURES/"_004.webp-5.png").is_file(), "Private comic fixtures are not distributed")
    def test_connected_white_text_outline_is_not_a_balloon(self):
        crop=Image.open(FIXTURES/'_004.webp-5.png').convert('RGB')
        im=Image.new('RGB',(crop.width+80,crop.height+80),'black');im.paste(crop,(40,40))
        r={'box':[40,40,40+crop.width,40+crop.height]}
        self.assertIsNone(BubbleMap(im,[r]).find(r))

if __name__=='__main__':unittest.main()
