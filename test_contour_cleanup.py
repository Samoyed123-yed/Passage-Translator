import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import engine
from typesetting import fit_adaptive,BubbleMap

class ContourCleanupTests(unittest.TestCase):
    def test_page_spanning_white_gutters_are_not_a_balloon(self):
        im=Image.new('RGB',(400,500),'white');d=ImageDraw.Draw(im)
        for y in range(40,470,80):d.rectangle((50,y,350,y+45),fill='black')
        self.assertIsNone(BubbleMap(im,[{'box':[10,80,45,230]}]).find({'box':[10,80,45,230]}))

    def test_partial_white_region_cannot_own_a_paragraph(self):
        im=Image.new('RGB',(400,500),'#888888');ImageDraw.Draw(im).rectangle((80,0,230,220),fill='white',outline='black',width=3)
        r={'box':[100,100,200,350]}
        self.assertIsNone(BubbleMap(im,[r]).find(r))
    def test_balloon_cut_by_top_edge_is_not_background(self):
        im=Image.new('RGB',(400,500),'#777777');d=ImageDraw.Draw(im)
        d.polygon([(100,0),(350,0),(350,400),(230,460),(80,380),(80,160)],fill='white',outline='black',width=3)
        r={'box':[130,50,300,360]}
        b=BubbleMap(im,[r]).find(r)
        self.assertIsNotNone(b)
        self.assertTrue(b['bordered'])
    def test_flat_cleaning_does_not_paint_outside_balloon(self):
        im=Image.new('RGB',(240,240),'#777777');d=ImageDraw.Draw(im)
        d.ellipse((30,30,210,210),fill='white',outline='black',width=3)
        safe=Image.new('L',im.size);ImageDraw.Draw(safe).ellipse((34,34,206,206),fill=1)
        body=np.asarray(safe)
        b={'box':[0,0,240,240],'safe':body,'body':body,'center':(120,120),'bordered':True}
        r={'id':0,'box':[30,30,210,210],'source':'明日はみんなで帰りましょう','translation':'明天一起回家','kind':'dialogue'}
        with patch('engine.BubbleMap.find',return_value=b):
            out=engine.render(im,Image.new('L',im.size,255),[r],engine.Vision(),'ja')
        self.assertTrue(np.array_equal(np.asarray(out)[body==0],np.asarray(im)[body==0]))

    def test_tall_original_remains_vertical(self):
        b={'box':[0,0,160,380],'safe':np.ones((380,160),np.uint8),'center':(80,190)}
        f=fit_adaptive('谢谢你快去吧',[30,50,130,320],True,engine.FONT,b,'ありがとうございます行きましょう',reference_size=30)
        self.assertTrue(f['vertical'])
        self.assertGreaterEqual(f['size'],30)

if __name__=='__main__':unittest.main()
