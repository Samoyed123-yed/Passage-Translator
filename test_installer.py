import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import setup_models as installer

class Response(io.BytesIO):
    status = 200
    headers = {}

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'public synthetic asset'
        self.item = {'file':'models/test.bin','url':'https://example.invalid/asset',
                     'sha256':hashlib.sha256(self.data).hexdigest(),'bytes':len(self.data)}

    def test_verified_existing_asset_never_downloads(self):
        p=self.root/self.item['file'];p.parent.mkdir();p.write_bytes(self.data)
        with patch.object(installer.urllib.request,'urlopen') as request:
            installer.fetch(self.root,self.item)
        request.assert_not_called()

    def test_corrupt_download_cannot_replace_existing_asset(self):
        p=self.root/self.item['file'];p.parent.mkdir();p.write_bytes(b'keep until replacement verified')
        with patch.object(installer.urllib.request,'urlopen',side_effect=lambda *a,**k:Response(b'corrupt')), patch.object(installer.time,'sleep'):
            with self.assertRaises(RuntimeError):installer.fetch(self.root,self.item)
        self.assertEqual(p.read_bytes(),b'keep until replacement verified')
        self.assertFalse(p.with_suffix('.bin.part').exists())

    def test_partial_download_resumes_and_validates(self):
        p=self.root/self.item['file'];p.parent.mkdir();p.with_suffix('.bin.part').write_bytes(self.data[:5])
        def response(req,**kwargs):
            self.assertEqual(req.headers['Range'],'bytes=5-')
            value=Response(self.data[5:]);value.status=206;value.headers={'Content-Range':f'bytes 5-{len(self.data)-1}/{len(self.data)}'}
            return value
        with patch.object(installer.urllib.request,'urlopen',side_effect=response):installer.fetch(self.root,self.item)
        self.assertEqual(p.read_bytes(),self.data)

    def test_server_without_range_restarts_download(self):
        p=self.root/self.item['file'];p.parent.mkdir();p.with_suffix('.bin.part').write_bytes(self.data[:5])
        with patch.object(installer.urllib.request,'urlopen',return_value=Response(self.data)):installer.fetch(self.root,self.item)
        self.assertEqual(p.read_bytes(),self.data)

    def test_verify_only_never_writes_or_downloads(self):
        with patch.object(installer.urllib.request,'urlopen') as request:
            with self.assertRaises(RuntimeError):installer.fetch(self.root,self.item,True)
        request.assert_not_called();self.assertEqual(list(self.root.iterdir()),[])

    def test_paths_cannot_escape_installation(self):
        for name in ('../outside.bin','/absolute','C:/elsewhere','models/../../escape','models\\..\\escape'):
            with self.subTest(name=name),self.assertRaises(ValueError):installer.target_path(self.root,name)

if __name__ == '__main__':unittest.main()
