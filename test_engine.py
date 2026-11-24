"""Offline integrity checks and synthetic multilingual end-to-end samples."""
import hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import engine

ROOT=Path(__file__).resolve().parent
def make_samples():
    examples=ROOT/'examples'; examples.mkdir(exist_ok=True)
    fonts={'en':'arialbd.ttf','ja':'YuGothM.ttc'}
    langs={'en':'英文','ja':'日文'}
    for lang,folder in langs.items():
        target=examples/folder; target.mkdir(exist_ok=True)
        image=Image.new('RGB',(1000,1400),'#e9e4dc'); d=ImageDraw.Draw(image)
        # Simple geometric panels, dialogue balloons, and narration on patterned artwork.
        d.rectangle((30,30,970,680),fill='#a8b8bf',outline='black',width=5)
        d.rectangle((30,710,970,1370),fill='#bcc4ad',outline='black',width=5)
        d.ellipse((90,70,480,570),fill='white',outline='black',width=3)
        d.ellipse((560,130,930,590),fill='white',outline='black',width=3)
        font=ImageFont.truetype(str(Path('C:/Windows/Fonts')/fonts[lang]),32)
        if lang=='en':
            d.multiline_text((155,185),'WE SHOULD\nLEAVE BEFORE\nTHE RAIN\nSTARTS.',font=font,fill='black',spacing=15)
            d.multiline_text((615,260),'WAIT FOR ME!\nI AM COMING\nWITH YOU.',font=font,fill='black',spacing=14)
            d.rectangle((100,800,890,950),fill='white',outline='black',width=2)
            d.multiline_text((135,835),'THAT EVENING, THE CITY\nWAS QUIET AGAIN.',font=font,fill='black',spacing=15)
            d.text((600,1150),'BOOM!',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',72),fill='black')
        elif lang=='ja':
            for x,text in [(350,'雨が降る前に'),(290,'帰りましょう。'),(800,'ちょっと待って！'),(740,'私も一緒に行く。')]:
                for y,c in enumerate(text): d.text((x,150+y*37),c,font=font,fill='black')
            d.rectangle((100,810,900,960),fill='white',outline='black',width=2)
            d.text((145,855),'その夜、街は静かだった。',font=font,fill='black')
        else:
            d.multiline_text((140,205),'비가 오기 전에\n집에 가자.',font=font,fill='black',spacing=18)
            d.multiline_text((600,265),'잠깐만 기다려!\n나도 같이 갈게.',font=font,fill='black',spacing=18)
            d.rectangle((100,810,900,960),fill='white',outline='black',width=2)
            d.text((145,855),'그날 밤 도시는 조용했다.',font=font,fill='black')
        image.save(target/'01-sample.jpg',quality=97)
    # Uppercase extension and nested same basename exercise real folder enumeration.
    nested=examples/'英文'/'第二页'; nested.mkdir(exist_ok=True)
    with Image.open(examples/'英文'/'01-sample.jpg') as im: im.save(nested/'01-sample.PNG')
    # Icon drawn as vector-style lettering, no generative media needed.
    icon=Image.new('RGB',(256,256),'#10151e'); draw=ImageDraw.Draw(icon)
    draw.rounded_rectangle((32,32,224,224),radius=38,fill='#658bff')
    draw.text((128,122),'P',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',150),anchor='mm',fill='#10151e')
    icon.save(ROOT/'passage.ico',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])

class Integrity(unittest.TestCase):
    def test_unsafe_output_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as t:
            p=Path(t)
            for destination in (p,p/'inside',p.parent):
                with self.assertRaises(ValueError): engine.validate_roots(p,destination)
    def test_natural_recursive_scan_and_extensions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as t:
            p=Path(t); (p/'chapter').mkdir()
            for name in ['10.JPG','2.png','not-image.txt','chapter/1.webp']: (p/name).touch()
            self.assertEqual([str(f.relative_to(p)).replace('\\','/') for f in engine.scan(p)],['2.png','10.JPG','chapter/1.webp'])
    def test_cannot_fit_leaves_original(self):
        self.assertIsNone(engine.layout('很长的译文'*200,[0,0,20,20],False))
    def test_external_network_blocked(self):
        engine.offline_guard()
        import socket
        with self.assertRaises(OSError): socket.getaddrinfo('example.com',443)

def integration(language):
    folders={'en':'英文','ja':'日文'}; source=ROOT/'examples'/folders[language]; output=source.with_name(source.name+'_中文')
    before={str(p):engine.sha256(p) for p in engine.scan(source)}
    started=time.monotonic()
    summary=engine.Batch(source,output,language,log=lambda s:print(s,flush=True)).run()
    records=[]
    for path in output.glob('.passage/**/*.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        with Image.open(record['source']) as a,Image.open(record['output']) as b:
            assert a.size==b.size
        assert engine.sha256(record['source'])==before[record['source']]
        records.append(record)
    result={'language':language,'seconds':round(time.monotonic()-started,2),'summary':summary,'records':records}
    engine.atomic_json(ROOT/'evidence'/f'integration-{language}.json',result)
    print(json.dumps({'language':language,'seconds':result['seconds'],'summary':summary},ensure_ascii=False),flush=True)
    if summary['failed'] or not records: raise RuntimeError('End-to-end test did not succeed')

if __name__=='__main__':
    if '--samples' in sys.argv: make_samples()
    elif '--integration' in sys.argv: integration(sys.argv[-1])
    else: unittest.main()
