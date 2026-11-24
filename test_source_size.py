import unittest
import numpy as np
from engine import FONT
from typesetting import fit_adaptive

class SourceSizeTests(unittest.TestCase):
    def test_indented_balloon_uses_original_location_before_shrinking(self):
        safe=np.zeros((300,300),np.uint8)
        safe[5:150,5:295]=1
        safe[150:295,140:160]=1
        bubble={'box':[0,0,300,300],'safe':safe,'center':(150,200),'bordered':True}
        f=fit_adaptive('明天见面',[100,40,200,120],False,FONT,bubble,'明日はここで会いましょう',reference_size=32)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f['size'],32)
        self.assertEqual(f.get('anchor'),'source-center')
        x,y,X,Y=f['bounds']
        self.assertTrue(safe[y:Y,x:X][np.asarray(f['tile'])>0].all())

    def test_short_floating_translation_stays_at_original_center(self):
        f=fit_adaptive('明天见',[20,20,220,140],False,FONT,None,'明日はここで会いましょう',reference_size=35)
        self.assertGreaterEqual(f['size'],35)
        x,y,X,Y=f['bounds']
        self.assertLessEqual(abs((x+X)/2-120),1)
        self.assertLessEqual(abs((y+Y)/2-80),1)
        self.assertTrue(x>=20 and y>=20 and X<=220 and Y<=140)

if __name__=='__main__':unittest.main()
