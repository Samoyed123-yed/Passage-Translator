"""Scanlation-style lettering (2026-11-20): centred blocks, word breaks, tight spacing."""
import unittest
import numpy as np
import fonts, linebreak, sakura
from typesetting import fit_balloon, PITCH, ADVANCE

FONT = fonts.pick(1)

def balloon(w=260, h=360):
    yy, xx = np.mgrid[0:h, 0:w]
    body = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1).astype(np.uint8)
    return {'box': [100, 100, 100 + w, 100 + h], 'safe': body, 'body': body, 'center': (100 + w / 2, 100 + h / 2), 'bordered': True}

class LineBreaks(unittest.TestCase):
    def test_words_are_not_split(self):
        self.assertEqual(linebreak.partition('哥哥和姐姐分开了', 2), ['哥哥和姐姐', '分开了'])
        self.assertEqual(linebreak.partition('因为他是姐姐的男朋友', 2), ['因为他是', '姐姐的男朋友'])

    def test_line_never_starts_with_particle_or_punctuation(self):
        for n in (2, 3):
            for line in linebreak.partition('没事的，很可爱的哦♥让我看看你可爱的地方♥', n)[1:]:
                self.assertNotIn(line[0], '的了吗呢吧，。！？♥')

    def test_break_after_de_is_avoided(self):
        self.assertEqual(linebreak.partition('毕竟你是姐姐的男朋友嘛', 2), ['毕竟你是', '姐姐的男朋友嘛'])

class CentredBalloon(unittest.TestCase):
    def test_block_is_centred_on_the_balloon(self):
        b = balloon(); f = fit_balloon('让哥哥多看看你可爱的地方吧♥', [150, 180, 310, 380], FONT, b, reference_size=26, source_text='お兄ちゃんにもっと見せて')
        cx = (f['bounds'][0] + f['bounds'][2]) / 2; cy = (f['bounds'][1] + f['bounds'][3]) / 2
        self.assertLess(abs(cx - 230) / 260, .04); self.assertLess(abs(cy - 280) / 360, .06)

    def test_size_follows_the_source_glyph(self):
        f = fit_balloon('还觉得不够吗？', [150, 180, 300, 380], FONT, balloon(), reference_size=30, source_text='まだ足りないの？')
        self.assertTrue(30 * .72 <= f['size'] <= 30 * 1.4, f['size'])

    def test_spacing_matches_scanlation_measurements(self):
        self.assertAlmostEqual(PITCH, 1.19, places=2); self.assertAlmostEqual(ADVANCE, 1.04, places=2)

    def test_emphasis_gets_its_own_larger_column(self):
        text = '早已成为哥哥最喜欢的小母狗了♥'
        span = sakura.locate('小母狗', text)
        f = fit_balloon(text, [150, 180, 310, 400], FONT, balloon(300, 420), reference_size=24, source_text='お兄ちゃんの大好きなメス犬♡', emphasis=span, emph_font=fonts.pick(2))
        self.assertIn('小母狗了♥', f['lines'])

class HorizontalBalloon(unittest.TestCase):
    def test_horizontal_lines_are_word_broken_and_centred(self):
        from typesetting import fit_adaptive
        w, h = 380, 240; yy, xx = np.mgrid[0:h, 0:w]
        body = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1).astype(np.uint8)
        b = {'box': [0, 0, w, h], 'safe': body, 'body': body, 'center': (w / 2, h / 2), 'bordered': True}
        f = fit_adaptive('你在说什么啊？我才没有生气呢！', [68, 77, 312, 163], False, FONT, b, 'What are you talking about?', (), 'auto', 24)
        self.assertFalse(f['vertical']); self.assertEqual(f['lines'], ['你在说什么啊？', '我才没有生气呢！'])
        cx = (f['bounds'][0] + f['bounds'][2]) / 2; cy = (f['bounds'][1] + f['bounds'][3]) / 2
        self.assertLess(abs(cx - w / 2) / w, .04); self.assertLess(abs(cy - h / 2) / h, .06)

class FontSwitch(unittest.TestCase):
    def test_rounded_family_is_installed_and_default(self):
        self.assertIn('rounded', fonts.available()); self.assertTrue(fonts.pick(1, 'rounded').is_file())

if __name__ == '__main__':
    unittest.main()
