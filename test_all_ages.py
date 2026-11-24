"""日常/成人 switch, handwritten notes and colored-text lettering (2026-11-24)."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import sakura
from direction import infer_direction
from text_rules import invalid_translation, normalize_preserved
from translation_profile import load_profile, relevant_terms, style_instructions


class AudienceTests(unittest.TestCase):
    def setUp(self): self.profile = load_profile()

    def test_general_books_skip_adult_glossary(self):
        text = '今日のオカズは唐揚げだ'
        adult = sakura.glossary_rows(self.profile, '', text, 'adult')
        general = sakura.glossary_rows(self.profile, '', text, 'general')
        self.assertTrue(any('撸管素材' in row for row in adult))
        self.assertFalse(any('撸管素材' in row for row in general))
        # Neutral entries stay in both modes.
        self.assertTrue(sakura.glossary_rows(self.profile, '', 'マジかよ', 'general'))

    def test_user_glossary_applies_in_both_modes(self):
        rows = sakura.glossary_rows(self.profile, 'オカズ->配菜', 'オカズ', 'general')
        self.assertEqual(rows, ['オカズ->配菜'])

    def test_style_rules_follow_the_switch(self):
        general, adult = style_instructions(self.profile, 'general'), style_instructions(self.profile, 'adult')
        self.assertIn('生活化', general); self.assertNotIn('露骨', general)
        self.assertIn('露骨', adult); self.assertNotIn('生活化', adult)
        auto = style_instructions(self.profile)
        self.assertIn('生活化', auto); self.assertIn('露骨', auto)
        self.assertNotIn('BLOWJOB', relevant_terms(self.profile, 'no blowjob here', 'general'))
        self.assertIn('BLOWJOB', relevant_terms(self.profile, 'no blowjob here', 'adult'))


class NoteTests(unittest.TestCase):
    def test_bullet_dot_is_not_untranslated_kana(self):
        self.assertFalse(invalid_translation('・好きな食べ物は目玉焼き', '・喜欢的食物是荷包蛋'))
        self.assertTrue(invalid_translation('好きな食べ物', '喜欢的たべもの'))

    def test_full_width_latin_title_may_stay_latin(self):
        self.assertFalse(invalid_translation('ＯＴＯＫＯＮＯＫＯ男の娘が好き', '喜欢OTOKONOKO男娘'))

    def test_latin_watermark_in_japanese_book_is_kept(self):
        r = {'source': '．．第６５０７ＡＧＨＩＺＥ．．', 'source_language': 'ja', 'kind': 'unknown', 'status': '待翻译'}
        self.assertTrue(normalize_preserved(r)); self.assertEqual(r['status'], '无需翻译，保留原文')
        r = {'source': 'ＯＴＯＫＯＮＯＫＯこういう男の娘が好き', 'source_language': 'ja', 'kind': 'unknown', 'status': '待翻译'}
        self.assertFalse(normalize_preserved(r))

    def test_wide_handwritten_rows_are_not_vertical_columns(self):
        # Three staggered rows of spaced glyphs: character gaps look like column
        # gutters and the rows touch, as in hand-lettered notes.
        mask = Image.new('L', (200, 100)); d = ImageDraw.Draw(mask)
        for row, y in enumerate((6, 36, 64)):
            for i in range(7):
                x = 8 + row * 4 + i * 26; d.rectangle((x, y + (i % 2) * 6, x + 17, y + 24 + (i % 2) * 6), fill=255)
        self.assertEqual(infer_direction(mask, 'ja')['orientation'], 'horizontal')

    def test_vertical_balloon_columns_stay_vertical(self):
        mask = Image.new('L', (90, 220)); d = ImageDraw.Draw(mask)
        for x in (10, 50):
            for j in range(6): d.rectangle((x, 10 + j * 34, x + 26, 36 + j * 34), fill=255)
        self.assertEqual(infer_direction(mask, 'ja')['orientation'], 'vertical')


if __name__ == '__main__':
    unittest.main()
