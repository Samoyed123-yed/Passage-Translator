"""Direction regressions: wide vertical text and tall horizontal text on one page."""
import json,sys,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import engine
from direction import infer_direction

ROOT=engine.ROOT/'evidence/direction-v1.1'

def sample():
    image=Image.new('RGB',(1400,1500),'white'); d=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/YuGothM.ttc',32)
    cases=[]
    def block(name,origin,texts,vertical,step):
        x,y=origin; boxes=[]
        for i,text in enumerate(texts):
            if vertical:
                for j,char in enumerate(text):
                    xy=(x-i*step,y+j*40); d.text(xy,char,font=font,fill='black'); boxes.append(d.textbbox(xy,char,font=font))
            else:
                xy=(x,y+i*step); d.text(xy,text,font=font,fill='black'); boxes.append(d.textbbox(xy,text,font=font))
        box=[min(b[0] for b in boxes)-3,min(b[1] for b in boxes)-3,max(b[2] for b in boxes)+3,max(b[3] for b in boxes)+3]
        cases.append({'name':name,'box':box,'expected':'vertical' if vertical else 'horizontal','expected_source':''.join(texts)})
    block('narrow vertical',(1160,100),['雨が降る前に','帰りましょう。'],True,60)
    block('tall horizontal',(120,100),['今日は雨','明日は晴','家で待つ','また会う','一緒に行','今は休む'],False,55)
    block('wide vertical floating',(1180,700),['雨が降る前に','家に帰ろう。','私も行くよ。','少し待って。'],True,60)
    block('wide horizontal floating',(100,750),['その夜、街は静かだった。','明日は友達に会います。'],False,60)
    mask=Image.fromarray((np.asarray(image.convert('L'))<150).astype(np.uint8)*255)
    return image,mask,cases

class Directions(unittest.TestCase):
    def test_per_region_direction_beats_rectangle_shape(self):
        image,mask,cases=sample()
        for c in cases:
            with self.subTest(case=c['name']):
                info=infer_direction(mask.crop(c['box']),'ja')
                self.assertEqual(info['orientation'],c['expected'],info)
                self.assertFalse(info['direction_uncertain'],info)
    def test_column_order_and_row_order(self):
        image,mask,cases=sample()
        for c in cases:
            info=infer_direction(mask.crop(c['box']),'ja'); lines=info['source_lines']
            values=[b[0] if c['expected']=='vertical' else b[1] for b in lines]
            self.assertEqual(values,sorted(values,reverse=c['expected']=='vertical'))
    def test_explicit_override(self):
        image,mask,cases=sample(); r=dict(cases[0],direction_override='horizontal')
        self.assertEqual(engine.set_direction(r,mask,'ja'),'horizontal')
        self.assertEqual(r['orientation'],'vertical')
    def test_old_records_reinfer_from_original_mask(self):
        image,mask,cases=sample(); r={'box':cases[2]['box']}
        self.assertEqual(engine.set_direction(r,mask,'ja'),'vertical')
    def test_render_coordinates_follow_requested_order(self):
        import typesetting
        real_draw=ImageDraw.Draw
        for direction in ('vertical','horizontal'):
            calls=[]
            class Recorder:
                def __init__(self,image): self.draw=real_draw(image)
                def text(self,xy,text,**kwargs): calls.append((xy,text)); return self.draw.text(xy,text,**kwargs)
            text='甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳'
            with patch.object(typesetting.ImageDraw,'Draw',Recorder):
                output=typesetting.make_tile(text,22,3,direction=='vertical',engine.FONT)
            self.assertEqual(''.join(t for xy,t in calls),text)
            if direction=='vertical':
                self.assertGreater(calls[0][0][0],calls[-1][0][0])
                for (a,_),(b,_) in zip(calls,calls[1:]):
                    if a[0]==b[0]: self.assertGreater(b[1],a[1])
                    else: self.assertLess(b[0],a[0]); self.assertEqual(b[1],calls[0][0][1])
            else:
                self.assertTrue(all(xy[0]==calls[0][0][0] for xy,t in calls))
                self.assertEqual([xy[1] for xy,t in calls],sorted(xy[1] for xy,t in calls))

def real_ocr():
    ROOT.mkdir(parents=True,exist_ok=True)
    image,mask,cases=sample(); image.save(ROOT/'original.png')
    vision=engine.Vision(); regions,detected_mask=vision.detect(image)
    for r in regions: vision.recognize(image,detected_mask,r,'ja')
    for case in cases:
        matches=[r for r in regions if engine.overlap_ratio(r['box'],case['box'])>.7]
        assert matches,('not detected',case)
        for r in matches: assert r['orientation']==case['expected'],(case,r)
        ordered=sorted(matches,key=lambda r:-r['box'][0] if case['expected']=='vertical' else r['box'][1])
        assert ''.join(r['source'] for r in ordered)==case['expected_source'],(case,ordered)
    engine.atomic_json(ROOT/'ocr.json',regions)
    print(json.dumps(regions,ensure_ascii=False,indent=2),flush=True)
    return image,detected_mask,regions

def full_test():
    image,mask,regions=real_ocr()
    engine.offline_guard(); translator=engine.LocalTranslator(lambda s:print(s,flush=True))
    try: translator.translate(regions,'ja','',[])
    finally: translator.close()
    output=engine.render(image,mask,regions,engine.Vision(),'ja',print)
    output.save(ROOT/'translated.png'); engine.atomic_json(ROOT/'result.json',regions)
    assert output.size==image.size
    assert all(r['status']=='已嵌字' for r in regions),regions
    compare=Image.new('RGB',(image.width*2,image.height));compare.paste(image);compare.paste(output,(image.width,0));compare.save(ROOT/'comparison.png')
    print('Mixed-direction OCR/translation/cleaning/rendering passed',flush=True)

if __name__=='__main__':
    if '--full' in sys.argv: full_test()
    elif '--ocr' in sys.argv: real_ocr()
    else: unittest.main()
