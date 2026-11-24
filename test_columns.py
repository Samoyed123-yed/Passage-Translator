"""Vertical Chinese follows the source columns: right-top first, no centered square block."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import engine, sakura
from typesetting import fit_columns, floating_space, _column_lengths

FONT = engine.FONT

class ColumnLayoutTests(unittest.TestCase):
    def test_short_translation_starts_at_top_right_of_source(self):
        fitted = fit_columns('明明知道这是坏事却还是兴奋得不得了', [100, 100, 210, 360], FONT, reference_size=22,
                             source_text='悪いことだってわかっていたのにすごく興奮した')
        self.assertIsNotNone(fitted)
        l, t, r, b = fitted['bounds']
        self.assertGreaterEqual(r, 200); self.assertLessEqual(t, 104)  # right edge, top edge
        self.assertGreater(b - t, 200)  # columns run down the region instead of a small square
        self.assertGreaterEqual(fitted['size'], 21)

    def test_size_stays_near_source_and_never_balloons(self):
        fitted = fit_columns('好', [0, 0, 200, 400], FONT, reference_size=20, source_text='はいはいはい')
        self.assertLessEqual(fitted['size'], 23)

    def test_stepped_caption_follows_plain_background(self):
        image = Image.new('RGB', (200, 300), (90, 90, 90)); d = ImageDraw.Draw(image)
        d.polygon([(80, 0), (200, 0), (200, 300), (0, 300), (0, 150), (80, 150)], fill='white')
        mask = Image.new('L', image.size)
        space = floating_space(image, mask, [0, 0, 200, 300])
        self.assertFalse(space[20, 20]); self.assertTrue(space[20, 150]); self.assertTrue(space[250, 20])
        fitted = fit_columns('一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十', [0, 0, 200, 300], FONT,
                             reference_size=22, source_text='あ' * 40, space=space)
        ink = np.zeros((300, 200), bool); l, t, r, b = fitted['bounds']
        ink[t:b, l:r] = np.asarray(fitted['tile']) > 0
        self.assertFalse(ink[:150, :78].any())  # nothing drawn on the artwork corner

    def test_sliver_beside_artwork_is_not_used_as_first_column(self):
        space = np.ones((300, 200), bool); space[:250, 150:] = False  # right strip only free at the bottom
        fitted = fit_columns('之后还要上课要是认真自慰的话', [0, 0, 200, 300], FONT, reference_size=24,
                             source_text='この後も授業あるのに本気オナニー', space=space)
        l, t, r, b = fitted['bounds']
        self.assertLessEqual(r, 152); self.assertLessEqual(t, 4)

    def test_balloon_columns_keep_source_length_first(self):
        safe = np.zeros((500, 300), np.uint8); safe[20:480, 20:280] = 1
        bubble = {'box': [0, 0, 300, 500], 'safe': safe, 'center': (150, 250)}
        fitted = fit_columns('多亏了这样鸡鸡一直都很湿润闪闪发光呢差不多也该', [60, 100, 240, 300], FONT, bubble,
                             reference_size=20, source_text='おかげでずっとちんこが潤っていっぱいツヤツヤですようもうそろそろ')
        l, t, r, b = fitted['bounds']
        self.assertLessEqual(b, 300 + 20); self.assertGreaterEqual(len(fitted['columns']), 2)

    def test_closing_punctuation_never_starts_a_column(self):
        text = '一二三四，五六七八。'
        lengths = _column_lengths(text, [4, 4, 4], 20, 22, [88, 88, 88])
        pos = 0
        for n in lengths:
            self.assertNotIn(text[pos], '，。'); pos += n

    def test_sakura_cleanup_keeps_loanword_lines_translatable(self):
        self.assertEqual(sakura.clean('这个大叔是个喜欢激烈play的人', 'プレイ'), '这个大叔是个喜欢激烈玩法的人')
        self.assertEqual(sakura.clean('我可能.喜欢.做爱呢', '私．セックス．好き'), '我可能……喜欢……做爱呢')
        self.assertEqual(sakura.clean('OK啦', 'OKだよ'), 'OK啦')

if __name__ == '__main__':
    unittest.main()
