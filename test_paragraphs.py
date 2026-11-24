import unittest
from PIL import Image,ImageDraw,ImageFont
import engine
from paragraphs import merge_paragraphs,english_lines,join_english
from text_rules import invalid_translation

def sample():
 im=Image.new('RGB',(440,560),'#999999');m=Image.new('L',im.size);d=ImageDraw.Draw(im);d.rectangle((50,40,380,500),fill='white',outline='black',width=2)
 f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30);boxes=[]
 for i,t in enumerate(['I ALSO','TOOK THE','LAST TRAIN','HOME BY','MYSELF.']):
  xy=(100,95+i*60);d.text(xy,t,font=f,fill='black');ImageDraw.Draw(m).text(xy,t,font=f,fill=255)
  a,b,c,e=d.textbbox(xy,t,font=f);boxes.append({'id':i,'box':[a-3,b-3,c+3,e+3],'source':'','translation':'','kind':'unknown'})
 return im,m,boxes

class Paragraphs(unittest.TestCase):
 def test_one_balloon_is_one_translation_unit(self):
  im,m,boxes=sample();groups=merge_paragraphs(im,m,boxes,'en');self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]['paragraph_parts']),5)
  b=groups[0]['box'];self.assertEqual(len(english_lines(im.crop(b),m.crop(b))),5)
 def test_border_separates_paragraphs(self):
  im,m,boxes=sample();ImageDraw.Draw(im).line((50,250,380,250),fill='black',width=3)
  groups=merge_paragraphs(im,m,boxes,'en');self.assertEqual(len(groups),2)
 def test_join_and_partial_english_rejection(self):
  self.assertEqual(join_english(['WE ARE','WAITING FOR','YOU.']),'WE ARE WAITING FOR YOU.')
  self.assertTrue(invalid_translation('I ALSO TOOK THE TRAIN.','我也 took the train.'))
  self.assertFalse(invalid_translation('HELLO JOHN.','你好，John。'))

if __name__=='__main__':unittest.main()
