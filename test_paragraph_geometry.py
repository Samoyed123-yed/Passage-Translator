import unittest
import numpy as np
from PIL import Image,ImageDraw
from paragraph_geometry import reconcile_regions
from text_style import prose_erase_mask

class ParagraphGeometryTests(unittest.TestCase):
    def test_crossing_glyph_expands_source_box_before_render(self):
        m=Image.new('L',(120,180));ImageDraw.Draw(m).rectangle((30,30,45,116),fill=255)
        r={'box':[25,25,60,110]}
        reconcile_regions([r],m,m)
        self.assertEqual(r['box'],[25,25,60,117])
        self.assertEqual(r['paragraph_geometry']['source_box'],r['box'])

    def test_shared_stroke_does_not_merge_two_paragraphs(self):
        m=Image.new('L',(120,120));ImageDraw.Draw(m).rectangle((35,30,65,90),fill=255)
        regions=[{'box':[20,20,50,100]},{'box':[50,20,80,100]}]
        reconcile_regions(regions,m,m)
        self.assertEqual([r['box'] for r in regions],[[20,20,50,100],[50,20,80,100]])

    def test_missing_line_support_cannot_expand_into_artwork(self):
        m=Image.new('L',(120,180));ImageDraw.Draw(m).rectangle((30,30,45,116),fill=255)
        r={'box':[25,25,60,110]}
        reconcile_regions([r],m,Image.new('L',m.size))
        self.assertEqual(r['box'],[25,25,60,110])

    def test_erase_mask_is_independent_of_neighbor_render_order(self):
        r={'box':[10,10,60,60]};other={'box':[70,10,110,60]};keep=Image.new('L',(120,120))
        first=prose_erase_mask(keep.size,(8,8,62,62),r,[r,other],keep)
        other['layout']={'bounds':[40,10,100,60]}
        second=prose_erase_mask(keep.size,(8,8,62,62),r,[r,other],keep)
        self.assertTrue(np.array_equal(first,second))

if __name__=='__main__':unittest.main()
