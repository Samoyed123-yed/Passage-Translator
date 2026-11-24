import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import engine
from task_storage import JobLock,commit_page,recover_commits
from automatic import save_matching
from visual_evidence import contour,effect_gate
from cleaning import planar_fill,MangaLama
from paragraph_geometry import merge_japanese

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.batch=SimpleNamespace(source=root/'source',output=root/'output',job=root/'job',state={'pages':{}})
        self.batch.source.mkdir();self.batch.output.mkdir()
        self.source=self.batch.source/'1.webp';Image.new('RGB',(48,40),'white').save(self.source)
        self.record={'source_sha256':engine.sha256(self.source)};self.target=self.batch.output/'1.webp'
    def tearDown(self):self.tmp.cleanup()
    def commit(self):
        return commit_page(self.batch,'1.webp',self.target,lambda p:save_matching(Image.new('RGB',(48,40),'red'),p),self.record,{'regions':[]})
    def test_numpy_values_in_regions_still_commit_under_final_name(self):
        # Regression: an int32 box from balloon-tail absorption made the JSON write fail,
        # leaving the translated page as .passage-stage-<hex> and the page "failed".
        regions=[{'box':[np.int32(3),np.int64(4),np.int32(20),np.int32(30)],'score':np.float32(.5)}]
        with JobLock(self.batch.output):
            commit_page(self.batch,'1.webp',self.target,lambda p:save_matching(Image.new('RGB',(48,40),'red'),p),self.record,{'regions':regions})
        self.assertTrue(self.target.is_file())
        self.assertEqual([p.name for p in self.batch.output.iterdir()],['1.webp'])
        saved=json.loads((self.batch.job/'working/1.webp.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['regions'][0]['box'],[3,4,20,30])
    def test_failed_commit_before_intent_leaves_no_stage_file(self):
        with JobLock(self.batch.output),patch.object(engine,'atomic_json',side_effect=TypeError('not serializable')):
            with self.assertRaises(TypeError):self.commit()
        self.assertEqual(list(self.batch.output.iterdir()),[])
    def test_crash_after_image_replace_finishes_metadata_without_rerender(self):
        actual=engine.atomic_json
        def fail(path,data):
            if Path(path).name=='manifest.json':raise OSError('simulated crash')
            actual(path,data)
        with JobLock(self.batch.output),patch.object(engine,'atomic_json',side_effect=fail):
            with self.assertRaises(OSError):self.commit()
        digest=engine.sha256(self.target);self.batch.state={'pages':{}}
        with JobLock(self.batch.output):recover_commits(self.batch)
        self.assertEqual(engine.sha256(self.target),digest)
        self.assertEqual(self.batch.state['pages']['1.webp']['output_sha256'],digest)
        self.assertFalse(list((self.batch.job/'pending').glob('*.json')))
        self.assertTrue((self.batch.job/'working/1.webp.json').is_file())
    def test_recovery_never_overwrites_external_edit(self):
        actual=engine.atomic_json
        def fail(path,data):
            if Path(path).name=='manifest.json':raise OSError('simulated crash')
            actual(path,data)
        with patch.object(engine,'atomic_json',side_effect=fail):
            with self.assertRaises(OSError):self.commit()
        self.target.write_bytes(b'USER EDIT')
        with self.assertRaises(RuntimeError):recover_commits(self.batch)
        self.assertEqual(self.target.read_bytes(),b'USER EDIT')
    def test_cross_process_lock_and_release(self):
        code='from task_storage import JobLock;import sys\nwith JobLock(sys.argv[1]):print("acquired")'
        args=[sys.executable,'-X','utf8','-c',code,str(self.batch.output)]
        with JobLock(self.batch.output):
            locked=subprocess.run(args,cwd=engine.ROOT,capture_output=True)
            self.assertNotEqual(locked.returncode,0)
        unlocked=subprocess.run(args,cwd=engine.ROOT,capture_output=True)
        self.assertEqual(unlocked.returncode,0,unlocked.stderr)
    @unittest.skipUnless(os.name=='nt','Windows Job Object')
    def test_windows_job_terminates_owned_child(self):
        from windows_job import ProcessJob
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            job=ProcessJob(child);job.close();self.assertIsNotNone(child.wait(timeout=5))
        finally:
            if child.poll() is None:child.kill();child.wait()

class VisionTests(unittest.TestCase):
    def test_missing_block_needs_independent_glyph_support(self):
        from visual_evidence import recover_text_regions
        mask=Image.new('L',(100,100));lines=mask.copy();hit={'class':2,'score':.95,'box':[10,10,70,90]}
        self.assertEqual(recover_text_regions([], [hit],mask,lines),[])
        for m in (mask,lines):ImageDraw.Draw(m).rectangle((20,20,30,75),fill=255)
        result=recover_text_regions([], [hit],mask,lines);self.assertEqual(len(result),1)
        self.assertEqual(len(recover_text_regions(result,[hit],mask,lines)),1)
    def test_gray_bubble_has_local_contour(self):
        image=Image.new('RGB',(320,260),'#777777');draw=ImageDraw.Draw(image)
        draw.ellipse((60,30,260,230),fill='#b6b6b6',outline='black',width=4)
        r={'box':[110,80,210,180],'visual_evidence':{'bubble_box':[58,28,263,233]}}
        b=contour(image,r,[r]);self.assertIsNotNone(b)
        self.assertTrue(b['bordered']);self.assertEqual(b['center'],(160,130))
        self.assertFalse(b['body'][0,0])
    def test_art_effect_needs_multiple_negative_signals(self):
        r={'source':'いくつ','ocr_confidence':.4,'visual_evidence':{'class':2,'line_support':.1}}
        self.assertTrue(effect_gate(r));self.assertTrue(r['skip_review'])
        r={'source':'いくつ','ocr_confidence':.4,'visual_evidence':{'class':1,'line_support':.1}}
        self.assertFalse(effect_gate(r))
        self.assertFalse(effect_gate({'source':'今日は一緒に帰りましょう','ocr_confidence':.1,'decorative_candidate':True}))
        self.assertFalse(effect_gate({'source':'いくつ','visual_evidence':{'class':None}}))
    def test_decoder_stops_loops_but_keeps_normal_repeat(self):
        from manga_translation.steps.text_recognition import _greedy_decode
        diagnostics={};ids=_greedy_decode(lambda ids:np.array([0,0,0,0,20]),250,diagnostics)
        self.assertTrue(diagnostics['repetition_stopped']);self.assertLess(len(ids),25)
        self.assertGreater(diagnostics['mean_probability'],.99)
        ids=_greedy_decode(lambda ids:np.array([0,0,0,20 if len(ids)>6 else 0,20 if len(ids)<=6 else 0]),50,{})
        self.assertEqual(len(ids),8)
    def test_japanese_pieces_merge_only_with_shared_text_evidence(self):
        image=Image.new('RGB',(200,240),'white');mask=Image.new('L',image.size)
        boxes=[[120,40,150,180],[80,40,110,180]]
        for b in boxes:ImageDraw.Draw(mask).rectangle(b,fill=255)
        parts=[{'box':b,'visual_evidence':{'class':1,'text_id':2,'text_box':[75,35,155,185]}} for b in boxes]
        merged=merge_japanese(image,mask,parts);self.assertEqual(len(merged),1);self.assertEqual(merged[0]['box'],[80,40,150,180])
        parts=[{'box':b} for b in boxes];self.assertEqual(len(merge_japanese(image,mask,parts)),2)
    def test_planar_gray_gradient_survives_interior_text_removal(self):
        y,x=np.mgrid[:100,:120];base=(80+x*.4+y*.2).astype(np.uint8);rgb=np.repeat(base[:,:,None],3,axis=2)
        original=Image.fromarray(rgb);image=original.copy();mask=Image.new('L',image.size)
        ImageDraw.Draw(image).rectangle((45,25,70,75),fill='black');ImageDraw.Draw(mask).rectangle((43,23,72,77),fill=255)
        filled=planar_fill(image,mask);self.assertIsNotNone(filled)
        self.assertLess(np.abs(np.asarray(filled).astype(float)-rgb).mean(),1.2)
    def test_native_lama_pads_to_eight_and_keeps_unmasked_pixels(self):
        calls=[]
        class Session:
            def run(self,_,feeds):
                calls.append(feeds['image'].shape);return [np.zeros_like(feeds['image'])]
        lama=MangaLama(None,lambda _:Session());im=Image.new('RGB',(171,137),'gray');mask=Image.new('L',im.size)
        ImageDraw.Draw(mask).rectangle((40,40,70,70),fill=255)
        out=lama.inpaint(im,mask);self.assertEqual(out.size,im.size)
        self.assertEqual(calls[0],(1,3,144,176));self.assertEqual(out.getpixel((5,5)),im.getpixel((5,5)));self.assertEqual(out.getpixel((50,50)),(0,0,0))
    def test_font_never_shrinks_below_floor(self):
        from typesetting import fit_adaptive
        self.assertIsNone(fit_adaptive('这是一段非常长的中文说明无法在这里容纳',[0,0,35,35],False,engine.FONT,reference_size=32))
        fitted=fit_adaptive('明天见',[0,0,150,150],True,engine.FONT,source_text='明日また会いましょう',reference_size=30)
        # 2026-11-20 scanlation measurement: Chinese is set at ~1.15-1.2x the source glyph.
        self.assertTrue(27<=fitted['size']<=37,fitted['size'])

class RecoveryTests(unittest.TestCase):
    def test_retranslation_exports_new_text_without_changing_old_records(self):
        import recovery_catalog,sakura
        from test_automatic import FakeTranslator
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'runtime/jobs'/('a'*24);(job/'working').mkdir(parents=True)
            manifest={'source':str(root/'missing-originals'),'output':str(root/'comic'),'pages':{}}
            engine.atomic_json(job/'manifest.json',manifest)
            meta=job/'working/1.webp.json';engine.atomic_json(meta,{'regions':[{'id':0,'source':'今日は雨です。','translation':'旧译文','kind':'dialogue'}]})
            before=meta.read_bytes()
            with patch.object(engine,'ROOT',root),patch.object(engine,'LocalTranslator',FakeTranslator),patch.object(engine,'offline_guard'),patch.object(sakura,'available',return_value=True):
                target=recovery_catalog.retranslate('a'*24,root/'text-report')
                self.assertEqual(meta.read_bytes(),before)
                data=json.loads(target.read_text(encoding='utf-8'));self.assertEqual(data[0]['items'][0]['old'],'旧译文')
                self.assertNotEqual(data[0]['items'][0]['new'],'旧译文')
                with self.assertRaises(FileExistsError):recovery_catalog.retranslate('a'*24,root/'text-report')
                with self.assertRaises(ValueError):recovery_catalog.retranslate('a'*24,root/'comic/subfolder')

if __name__=='__main__':unittest.main()
