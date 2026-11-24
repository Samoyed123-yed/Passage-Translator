"""User rule (2026-11-16): cries in a drawn balloon are translated; cries and sound
effects outside balloons are left exactly as drawn (not translated, not erased)."""
import unittest
from automatic import in_balloon, outside_balloon_effect, mark_effect, expected_preservation

def region(source, **extra):
    return dict({'source': source, 'kind': 'dialogue', 'status': '待翻译'}, **extra)

class BalloonEffectRules(unittest.TestCase):
    def test_floating_short_kana_effects_are_left_alone(self):
        for text in ('ずにょ', 'とん', 'ぴく', 'てぶくっ', 'きゅん', 'おっおっ♡', 'あぁ♥', 'ごっくんっ'):
            self.assertTrue(outside_balloon_effect(region(text)), text)

    def test_same_cry_inside_a_balloon_is_translated(self):
        for evidence in ({'class': 1}, {'bubble_box': [0, 0, 9, 9]}):
            self.assertFalse(outside_balloon_effect(region('あっ♡', visual_evidence=evidence)))
        self.assertTrue(in_balloon(region('あっ♡', boxed_voice=True)))

    def test_floating_narration_and_kanji_dialogue_still_translate(self):
        self.assertFalse(outside_balloon_effect(region('だけど', visual_evidence={'class': 2})))
        self.assertFalse(outside_balloon_effect(region('出して♥')))
        self.assertFalse(outside_balloon_effect(region('悪いことだってわかっていたのに')))

    def test_unreadable_floating_art_skips_ocr_retry(self):
        self.assertTrue(outside_balloon_effect(region('ぬくのやべゞ'), unreadable=True))
        self.assertFalse(outside_balloon_effect(region('ぬくのやべゞ', visual_evidence={'class': 1}), unreadable=True))

    def test_left_effect_is_an_expected_skip_not_a_failure(self):
        r = region('ずにょ', automatic_reason='识别重试后仍不明确'); mark_effect(r)
        self.assertEqual(r['kind'], 'sfx'); self.assertTrue(expected_preservation(r))

if __name__ == '__main__':
    unittest.main()
