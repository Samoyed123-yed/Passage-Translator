import unittest
import numpy as np
from engine import FONT
from typesetting import make_tile,fit_adaptive

class ReferenceLayoutTests(unittest.TestCase):
    def test_horizontal_rows_keep_clear_gaps(self):
        for text in ('明天我们一起出发请记得带上雨伞','明天ABC一起出发♥请记得带上雨伞'):
            tile=make_tile(text,36,3,False,FONT)
            rows=np.asarray(tile).max(axis=1)>0
            edges=np.diff(np.r_[False,rows,False].astype(int))
            starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            self.assertEqual(len(starts),3)
            self.assertTrue(np.all(starts[1:]-ends[:-1]>=3))
            self.assertLess(tile.height,36*1.32*3)

    def test_sparse_bubble_keeps_obstacle_clear(self):
        safe=np.ones((260,320),np.uint8)
        b={'box':[0,0,320,260],'center':(160,130),'safe':safe,'bordered':True}
        f=fit_adaptive('明天一起出发吧',[120,100,200,160],False,FONT,b,'明天一起出发吧',[(0,0,320,65)])
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f['bounds'][1],65)
        self.assertLessEqual(f['bounds'][2],320)
        self.assertLessEqual(f['bounds'][3],260)

if __name__=='__main__':unittest.main()
