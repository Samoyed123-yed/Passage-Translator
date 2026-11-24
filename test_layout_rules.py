"""Lettering rules from the all-ages test books (2026-11-24)."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import fonts
import linebreak
import sakura
from paragraph_geometry import split_staggered
from text_rules import invalid_translation, normalize_preserved
from typesetting import BubbleMap, fit_balloon


def two_groups():
    """Right group high, left group low, 30 px glyphs, like a stepped caption."""
    mask = Image.new('L', (220, 250)); d = ImageDraw.Draw(mask)
    for x in (120, 160): d.rectangle((x, 10, x + 29, 149), fill=255)
    for x in (20, 60): d.rectangle((x, 90, x + 29, 229), fill=255)
    return {'box': [0, 0, 220, 250], 'orientation': 'vertical', 'source_font_size': 30}, mask


class SplitTests(unittest.TestCase):
    def test_joined_boxes_split_into_one_paragraph_each(self):
        r, mask = two_groups()
        gray = np.full((250, 220), 255, np.uint8)
        gray[70:73, 10:100] = 0    # edge of the lower box, just above its text
        gray[170:173, 110:200] = 0  # edge of the upper box, just below its text
        pieces = split_staggered(r, mask, gray)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(p['balloon_split'] for p in pieces))

    def test_staggered_columns_in_one_balloon_stay_together(self):
        r, mask = two_groups()
        self.assertEqual(len(split_staggered(r, mask, np.full((250, 220), 255, np.uint8))), 1)


class SizeTests(unittest.TestCase):
    def test_chinese_no_longer_than_japanese_keeps_the_source_size(self):
        fitted = fit_balloon('我的推就在我家', (0, 0, 80, 330), fonts.pick(1), None, (), 40, 'わたしのおしがじたくに')
        self.assertIsNotNone(fitted)
        self.assertGreaterEqual(fitted['size'], 40)


class LobeTests(unittest.TestCase):
    def test_two_lobed_balloon_is_lettered_right_high_left_low(self):
        import cv2
        from typesetting import fit_lobes
        safe = np.zeros((700, 600), np.uint8)
        cv2.ellipse(safe, (400, 220), (170, 200), 0, 0, 360, 1, -1)
        cv2.ellipse(safe, (200, 470), (170, 200), 0, 0, 360, 1, -1)
        bubble = {'box': [0, 0, 600, 700], 'safe': safe.astype(bool), 'center': (300, 350)}
        f = fit_lobes('我真的不知道该怎么办才好啊你说呢', [[340, 90, 460, 350], [140, 340, 260, 600]], fonts.pick(1), bubble,
                      'どうすればいいのかわからないよねえどうしよう', (), 40)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f['size'], 40)
        self.assertEqual(f['anchor'], 'two-lobes')
        ink = np.asarray(f['tile']) > 0
        x0, y0 = f['position']
        self.assertTrue(ink[:max(1, 300 - y0), max(0, 330 - x0):].any())   # upper-right lobe used
        self.assertTrue(ink[max(0, 420 - y0):, :max(1, 270 - x0)].any())   # lower-left lobe used


class BalloonTests(unittest.TestCase):
    def test_black_thought_balloon_is_found(self):
        image = Image.new('RGB', (400, 400), 'white'); d = ImageDraw.Draw(image)
        d.ellipse((140, 120, 260, 280), fill='black')
        r = {'box': [175, 150, 225, 250]}
        found = BubbleMap(image, [r]).find(r)
        self.assertIsNotNone(found)
        self.assertLessEqual(found['box'][2] - found['box'][0], 125)

    def test_panel_background_reaching_the_page_edge_is_not_a_balloon(self):
        image = Image.new('RGB', (400, 400), 'white'); d = ImageDraw.Draw(image)
        d.rectangle((-5, -5, 300, 300), outline='black', width=4)
        r = {'box': [100, 100, 140, 150]}
        self.assertIsNone(BubbleMap(image, [r]).find(r))


class ScreentoneTests(unittest.TestCase):
    def test_halftone_joined_through_a_gap_is_not_balloon_interior(self):
        image = Image.new('RGB', (1200, 1200), 'white'); d = ImageDraw.Draw(image)
        for y in range(30, 570, 9):  # dotted screen inside a framed panel
            for x in range(30, 570, 9): d.ellipse((x, y, x + 4, y + 4), fill='black')
        d.rectangle((20, 20, 580, 580), outline='black', width=6)
        d.ellipse((150, 150, 400, 450), fill='white', outline='black', width=5)
        d.rectangle((270, 140, 280, 160), fill='white')  # gap in the outline
        r = {'box': [240, 220, 310, 380]}
        found = BubbleMap(image, [r]).find(r)
        self.assertIsNotNone(found)
        self.assertLessEqual(found['body'].sum(), 3.14 * 125 * 150 * 1.15)


class BackgroundSplitTests(unittest.TestCase):
    def test_balloon_column_beside_white_on_black_shout_is_separate(self):
        from paragraph_geometry import split_by_background
        gray = np.full((300, 300), 255, np.uint8); gray[:, :150] = 0
        mask = Image.new('L', (300, 300)); d = ImageDraw.Draw(mask)
        g = np.array(gray)
        for y in range(10, 290, 70):  # big white glyphs on black, paper between them
            d.rectangle((30, y, 120, y + 55), fill=255); g[y:y + 56, 30:121] = 255
        for y in range(40, 200, 32):  # small black glyphs in the white balloon
            d.rectangle((190, y, 220, y + 24), fill=255); g[y:y + 25, 190:221] = 0
        pieces, light = split_by_background({'box': [0, 0, 300, 300], 'orientation': 'vertical'}, mask, g)
        self.assertEqual(len(pieces), 2)
        self.assertGreater(light['box'][0], 150)


class TextTests(unittest.TestCase):
    def test_affirmative_dewa_aru_is_not_read_as_negative(self):
        from text_rules import sakura_input
        self.assertEqual(sakura_input('でも発売日に買うほど好きではある：'), 'でも発売日に買うほど好きだ：')
        self.assertEqual(sakura_input('ここではある程度'), 'ここではある程度')

    def test_spaced_dot_ellipsis_becomes_chinese_ellipsis(self):
        self.assertEqual(sakura.clean('被看光了.  .  .  .', 'えっ'), '被看光了……')

    def test_filler_word_is_speech_not_effect(self):
        from automatic import outside_balloon_effect
        self.assertFalse(outside_balloon_effect({'source': 'あの', 'box': [0, 0, 10, 10]}))
        self.assertTrue(outside_balloon_effect({'source': 'ドキ', 'box': [0, 0, 10, 10]}))

    def test_publisher_footer_is_kept(self):
        r = {'source': '無断転載禁止・生成ＡＩ学習禁止', 'translation': '禁止擅自转载', 'kind': 'unknown', 'status': '待翻译'}
        self.assertTrue(normalize_preserved(r))

    def test_kanji_name_plate_is_kept(self):
        r = {'source': '依野＿真夜', 'translation': '依野＿真夜', 'kind': 'unknown', 'status': '待翻译'}
        self.assertTrue(normalize_preserved(r))
        r = {'source': '依野', 'translation': '依野先生', 'kind': 'unknown', 'status': '待翻译'}
        self.assertFalse(normalize_preserved(r))

    def test_big_art_descriptor_and_repeated_kanji_are_kept(self):
        r = {'source': '華奢な', 'translation': '', 'kind': 'unknown', 'status': '待翻译',
             'box': [2115, 1670, 2628, 1873], 'ocr_reading_order': 'large-line-rescaled'}
        self.assertTrue(normalize_preserved(r))
        shout = {'source': 'すっごいすっごい', 'translation': '', 'kind': 'unknown', 'status': '待翻译', 'box': [269, 300, 642, 591], 'ocr_reading_order': 'large-line-rescaled'}
        self.assertFalse(normalize_preserved(shout))
        self.assertTrue(normalize_preserved({'source': '男男男男男男男男実', 'translation': '', 'kind': 'unknown', 'status': '待翻译'}))

    def test_names_are_not_broken_across_lines(self):
        linebreak.protect(linebreak.names_in('天咲ひかりです！', '我是天咲光！'))
        self.assertEqual(linebreak.partition('我是天咲光！', 2), ['我是', '天咲光！'])

    def test_stray_kana_and_colon_ellipsis(self):
        self.assertEqual(sakura.clean('那是……？シ', 'それは…？し'), '那是……？')
        self.assertTrue(sakura.clean('不过我很喜欢：', '好きではある：').endswith('……'))

    def test_short_interjection_may_become_a_chinese_interjection(self):
        self.assertFalse(invalid_translation('へえ．．．', '哦……'))
        self.assertFalse(invalid_translation('ひかり', '光'))


if __name__ == '__main__':
    unittest.main()
