import unittest
from PIL import Image,ImageDraw
from paragraph_geometry import source_ink_size
from typesetting import fit_adaptive
from engine import FONT

class SourceInkTests(unittest.TestCase):
    def test_eroded_detector_core_does_not_halve_vertical_font(self):
        r={'box':[0,0,95,142],'orientation':'vertical','source_lines':[[60,13,95,59],[35,13,59,121],[7,13,31,122]],
           'paragraph_geometry':{'det_lines':[[40,20,51,116],[14,19,23,117],[62,18,83,57]]}}
        size,method=source_ink_size(r,Image.new('L',(95,142)))
        self.assertEqual(size,20)
        self.assertEqual(method,'full-source-ink-lines')
        layout=fit_adaptive('明天大家一起出门吧',[0,0,95,142],True,FONT,source_text='明日はみんなで一緒に出かけましょう',reference_size=size)
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(layout['size'],18)
        self.assertGreater(layout['bounds'][3]-layout['bounds'][1],45)

    def test_horizontal_height_comes_from_full_ink(self):
        r={'box':[0,0,210,80],'orientation':'horizontal','source_lines':[[4,4,180,32],[4,42,150,70]]}
        self.assertEqual(source_ink_size(r,Image.new('L',(210,80)))[0],24)

    def test_merged_projection_does_not_turn_four_columns_into_one_huge_glyph(self):
        mask=Image.new('L',(130,180));draw=ImageDraw.Draw(mask)
        cores=[]
        for x in (8,34,60,86):
            draw.rectangle((x,12,x+20,152),fill=255)
            cores.append([x+6,20,x+15,145])
        r={'box':[0,0,120,170],'orientation':'vertical','source_lines':[[4,8,110,158]],'paragraph_geometry':{'det_lines':cores}}
        size,method=source_ink_size(r,mask)
        self.assertEqual(size,21)
        self.assertEqual(method,'full-ink-per-detector-line')

if __name__=='__main__':unittest.main()
