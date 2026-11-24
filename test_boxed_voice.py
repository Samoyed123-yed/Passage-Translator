import unittest
from PIL import Image,ImageDraw
from text_rules import mark_boxed_voice,normalize_preserved
from automatic import unresolved_count

class BoxedVoiceTests(unittest.TestCase):
    def image(self):
        im=Image.new('RGB',(400,400),'#888888')
        ImageDraw.Draw(im).ellipse((60,60,340,340),fill='white',outline='black',width=5)
        return im

    def test_voice_inside_balloon_stays_translatable(self):
        r={'box':[140,130,260,250],'source':'あああ！','kind':'sfx','status':'跳过特效字'}
        self.assertTrue(mark_boxed_voice(self.image(),[r],r))
        self.assertFalse(normalize_preserved(r))
        self.assertEqual(r['kind'],'dialogue')

    def test_floating_voice_preserved_and_not_reported(self):
        r={'box':[10,10,50,50],'source':'あああ！','automatic_reason':'旧的识别失败'}
        self.assertFalse(mark_boxed_voice(self.image(),[r],r))
        self.assertTrue(normalize_preserved(r))
        self.assertEqual(unresolved_count([r]),0)

    def test_regular_sentence_failure_remains_visible(self):
        r={'box':[140,130,260,250],'source':'明日はどこへ行くの？','status':'识别不确定，保留原文'}
        self.assertTrue(mark_boxed_voice(self.image(),[r],r))
        self.assertTrue(r['boxed_prose'])
        self.assertFalse(normalize_preserved(r))
        self.assertEqual(unresolved_count([r]),1)

    def test_unlisted_drawn_out_voice_enters_translation(self):
        r={'box':[140,130,260,250],'source':'ふい～～っ','kind':'sfx'}
        self.assertTrue(mark_boxed_voice(self.image(),[r],r))
        self.assertTrue(r['boxed_voice'])
        self.assertFalse(normalize_preserved(r))

    def test_background_effect_quiet_but_floating_sentence_not_hidden(self):
        r={'box':[10,10,50,50],'source':'ガクガク','automatic_reason':'OCR重试失败'}
        self.assertFalse(mark_boxed_voice(self.image(),[r],r))
        self.assertTrue(normalize_preserved(r))
        self.assertEqual(unresolved_count([r]),0)
        prose={'box':[10,10,50,50],'source':'明日は雨です','status':'需检查译文'}
        self.assertFalse(mark_boxed_voice(self.image(),[prose],prose))
        self.assertFalse(normalize_preserved(prose))
        self.assertEqual(unresolved_count([prose]),1)

    def test_words_mixed_with_shout_are_not_effects(self):
        r={'box':[140,130,260,250],'source':'おおお～走るう～～','kind':'sfx'}
        self.assertTrue(mark_boxed_voice(self.image(),[r],r))
        self.assertTrue(r['boxed_prose'])
        self.assertEqual(r['kind'],'dialogue')

if __name__=='__main__':unittest.main()
