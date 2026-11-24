import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import automatic,engine
from text_style import original_ink,recover_flat_mask
from typesetting import fit_text,fit_adaptive

class Styles(unittest.TestCase):
    def make(self,color,bg='white'):
        im=Image.new('RGB',(320,160),bg);mask=Image.new('L',im.size)
        font=ImageFont.truetype(str(engine.FONT),36)
        for target,fill in ((im,color),(mask,255)):ImageDraw.Draw(target).text((30,45),'HELLO',font=font,fill=fill)
        return im,mask,[26,45,165,92]
    def test_original_solid_colors(self):
        for color in ((0,0,0),(255,255,255),(163,45,190),(210,35,45),(30,155,70),(40,75,205)):
            im,mask,box=self.make(color,'#222222' if color==(255,255,255) else 'white')
            actual=original_ink(im,mask,box)
            self.assertLess(max(abs(a-b) for a,b in zip(color,actual)),15,(color,actual))
    def test_black_ink_on_colored_background_stays_black(self):
        im,mask,box=self.make('black','#b9d8e6');self.assertEqual(original_ink(im,mask,box),(0,0,0))
    def test_render_uses_original_color(self):
        im,mask,box=self.make((163,45,190));r={'id':0,'source':'HELLO','box':box,'translation':'你好','kind':'dialogue'}
        class Plain:
            def inpaint(self,im,mask):return Image.new('RGB',im.size,'white')
        result=engine.render(im,mask,[r],Plain(),'en',automatic=True)
        self.assertEqual(r['layout']['color'],[163,45,190])
        self.assertTrue((np.asarray(result)==[163,45,190]).all(2).any())
    def test_small_text_enlarges_only_with_room(self):
        b={'box':[0,0,300,300],'center':(150,150),'safe':np.ones((300,300),np.uint8)};box=[115,115,185,185];text='明天一起回家吧'
        old=fit_text(text,box,False,engine.FONT,b,text)
        new=fit_adaptive(text,box,False,engine.FONT,b,text)
        self.assertGreater(new['size'],old['size']);self.assertLess(new['occupancy'],.5)
    def test_target_direction_preserves_source_reading_order(self):
        for w,h,source_vertical,wanted in ((440,100,True,True),(90,420,True,True),(90,420,False,False)):
            b={'box':[0,0,w,h],'center':(w/2,h/2),'safe':np.ones((h,w),np.uint8)}
            f=fit_adaptive('我们明天一起在门口集合。',[w//2-25,h//2-25,w//2+25,h//2+25],source_vertical,engine.FONT,b)
            self.assertEqual(f['vertical'],wanted)
            self.assertTrue(0<=f['bounds'][0]<f['bounds'][2]<=w and 0<=f['bounds'][1]<f['bounds'][3]<=h)

class FakeVision:
    def detect(self,image):
        self.source_pixels=image.tobytes();self.source_size=image.size
        return [{'id':0,'box':[15,15,130,78],'source':'','translation':'','kind':'dialogue'}],Image.fromarray((np.asarray(image.convert('L'))<130).astype(np.uint8)*255)
    def recognize(self,image,mask,r,lang):
        rendered=image.size==getattr(self,'source_size',None) and image.tobytes()!=self.source_pixels
        r.update(source='我们赶在下雨前回家。' if rendered else 'We should go home before the rain.',status='待翻译',kind='dialogue');return True
    def inpaint(self,image,mask):return Image.new('RGB',image.size,'white')
class FakeTranslator:
    def __init__(self,log):pass
    def translate(self,regions,*args):
        for r in regions:r.update(translation='我们赶在下雨前回家。',kind='dialogue',status='已翻译')
    def close(self):pass
class UncertainTranslator(FakeTranslator):
    def translate(self,regions,*args):
        for r in regions:r.update(translation='',kind='uncertain',status='需检查译文')

class AutomaticWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=engine.ROOT/'runtime');self.root=Path(self.temp.name);self.src=self.root/'Book';self.src.mkdir()
        self.patches=[patch.object(automatic,'CACHE_ROOT',self.root/'cache'),patch.object(engine,'Vision',FakeVision),patch.object(engine,'LocalTranslator',FakeTranslator),patch.object(engine,'model_status',return_value=[])]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def picture(self,name='1.webp'):
        path=self.src/name;path.parent.mkdir(parents=True,exist_ok=True);im=Image.new('RGB',(180,120),'white')
        ImageDraw.Draw(im).text((20,25),'HELLO',font=ImageFont.truetype(str(engine.FONT),24),fill='black');im.save(path);return path
    def test_original_names_tree_formats_clean_output_and_resume(self):
        originals=[self.picture(n) for n in ('1.webp','chapter/2.PNG','chapter/3.jpg')];(self.src/'empty').mkdir();(self.src/'note.txt').write_text('original note')
        hashes={str(p):engine.sha256(p) for p in originals};batch=automatic.AutoBatch(self.src,language='en',log=lambda _:None);result=batch.run()
        self.assertEqual(result['completed'],3);self.assertEqual(result['failed'],0)
        self.assertEqual(batch.output,self.src.with_name('Book（中译）'))
        self.assertEqual({str(p.relative_to(batch.output)) for p in batch.output.rglob('*') if p.is_file()},{'1.webp',str(Path('chapter/2.PNG')),str(Path('chapter/3.jpg')),'note.txt'})
        self.assertTrue((batch.output/'empty').is_dir());self.assertTrue((batch.job/'working').exists())
        for p in originals:
            self.assertEqual(engine.sha256(p),hashes[str(p)])
            with Image.open(p) as a,Image.open(batch.output/p.relative_to(self.src)) as b:self.assertEqual(a.size,b.size);self.assertEqual(a.format,b.format)
        with patch.object(FakeVision,'detect',side_effect=AssertionError('Resume must not reprocess')):
            resumed=automatic.AutoBatch(self.src,language='en',log=lambda _:None).run()
        self.assertEqual(resumed['resumed'],3)
    def test_automatic_fallback_keeps_original_bytes(self):
        source=self.picture()
        with patch.object(engine,'LocalTranslator',UncertainTranslator):
            b=automatic.AutoBatch(self.src,language='en',log=lambda _:None);s=b.run()
        self.assertEqual(s['failed'],0);self.assertEqual(s['review'],0);self.assertEqual(s['retained_pages'],1)
        self.assertEqual(source.read_bytes(),(b.output/source.name).read_bytes());self.assertTrue((b.job/'working').exists())
    def test_saved_unresolved_page_requires_explicit_reprocess(self):
        self.picture()
        with patch.object(engine,'LocalTranslator',UncertainTranslator):
            first=automatic.AutoBatch(self.src,language='en',log=lambda _:None).run()
        self.assertEqual(first['retained_pages'],1)
        second=automatic.AutoBatch(self.src,language='en',log=lambda _:None).run()
        self.assertEqual(second['resumed'],1);self.assertEqual(second['completed'],0)
        third=automatic.AutoBatch(self.src,language='en',log=lambda _:None,reprocess=True).run()
        self.assertEqual(third['completed'],1);self.assertEqual(third['resumed'],0)

    def test_page_recovery_is_bounded_and_not_reported_as_success(self):
        self.picture()
        with patch.object(engine,'LocalTranslator',UncertainTranslator):
            batch=automatic.AutoBatch(self.src,language='en',log=lambda _:None)
            # 2026-11-16: no whole-page second pass; the page runs once and the
            # leftover text is still reported, not counted as success.
            with patch.object(batch,'finish',wraps=batch.finish) as finish:
                result=batch.run()
                self.assertEqual(finish.call_count,1)
        self.assertEqual(result['retained_pages'],1)
    def test_cancel_keeps_checkpoint_and_resume_cleans_it(self):
        self.picture('1.webp');self.picture('2.webp');stop=[False]
        def event(*args):
            if args[0]=='done':stop[0]=True
        b=automatic.AutoBatch(self.src,language='en',log=lambda _:None,event=event,cancel=lambda:stop[0]);s=b.run()
        self.assertTrue(s['cancelled']);self.assertTrue((b.job/'working').exists());self.assertFalse((b.output/'2.webp').exists())
        resumed=automatic.AutoBatch(self.src,language='en',log=lambda _:None).run();self.assertEqual(resumed['resumed'],1);self.assertEqual(resumed['completed'],1);self.assertTrue((b.job/'working').exists())
    def test_unrelated_folder_gets_a_new_name(self):
        self.picture();existing=self.src.with_name('Book（中译）');existing.mkdir();(existing/'personal.txt').write_text('keep')
        b=automatic.AutoBatch(self.src,language='en',log=lambda _:None);self.assertEqual(b.output.name,'Book（中译 2）');self.assertEqual(b.run()['failed'],0);self.assertEqual((existing/'personal.txt').read_text(),'keep')
    def test_modified_result_is_not_overwritten(self):
        self.picture();b=automatic.AutoBatch(self.src,language='en',log=lambda _:None);b.run();out=b.output/'1.webp';out.write_bytes(b'USER EDIT')
        s=automatic.AutoBatch(self.src,language='en',log=lambda _:None).run();self.assertEqual(s['failed'],1);self.assertEqual(out.read_bytes(),b'USER EDIT')
    def test_multiframe_file_keeps_every_frame_and_original_bytes(self):
        source=self.src/'animation.gif';frames=[Image.new('RGB',(80,60),c) for c in ('red','blue')]
        frames[0].save(source,save_all=True,append_images=frames[1:],duration=80,loop=0)
        b=automatic.AutoBatch(self.src,language='en',log=lambda _:None);s=b.run()
        self.assertEqual(s['failed'],0);self.assertEqual(s['retained_pages'],0);self.assertEqual(source.read_bytes(),(b.output/source.name).read_bytes())
        with Image.open(b.output/source.name) as im:self.assertEqual(im.n_frames,2)
    def test_dot_chapters_included_but_passage_metadata_excluded(self):
        self.picture('.chapter/1.webp');self.picture('.passage/mask.png')
        b=automatic.AutoBatch(self.src,language='en',log=lambda _:None);s=b.run()
        self.assertEqual(s['completed'],1);self.assertTrue((b.output/'.chapter/1.webp').exists());self.assertFalse((b.output/'.passage').exists())
    def test_working_cleanup_refuses_paths_outside_cache(self):
        work=self.src/'working';work.mkdir();(work/'user.txt').write_text('KEEP')
        with self.assertRaises(ValueError):automatic.remove_working(self.src)
        self.assertEqual((work/'user.txt').read_text(),'KEEP')

if __name__=='__main__':unittest.main()
