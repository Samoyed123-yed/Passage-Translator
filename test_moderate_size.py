import unittest
import numpy as np
from PIL import Image
from engine import FONT
from typesetting import fit_adaptive
from text_style import prose_erase_mask

class ModerateSizeTests(unittest.TestCase):
    def test_shorter_translation_does_not_expand_to_fill(self):
        b={'box':[0,0,300,300],'safe':np.ones((300,300),np.uint8),'center':(150,150),'bordered':True}
        f=fit_adaptive('明天见',[60,60,240,240],True,FONT,b,'明日はここで会いましょう',reference_size=32)
        # 2026-11-20: short replies in balloons are set larger (up to 1.4x), as the scanlation group does.
        self.assertTrue(32<=f['size']<=45,f['size'])

    def test_complete_floating_columns_leave_empty_corner(self):
        r={'box':[20,20,120,180],'source_lines':[[0,0,25,160],[75,60,100,160]],'direction_uncertain':False}
        mask=prose_erase_mask((200,200),(17,17,123,183),r,[r],Image.new('L',(200,200)))
        self.assertEqual(mask.getpixel((100-17,30-17)),0)
        self.assertEqual(mask.getpixel((30-17,30-17)),255)

if __name__=='__main__':unittest.main()
