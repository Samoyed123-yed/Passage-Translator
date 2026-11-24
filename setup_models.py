"""Explicit online installation only. Never imported by the translation engine."""
from pathlib import Path, PurePosixPath
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def target_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in relative or '\\' in relative:
        raise ValueError(f'Unsafe asset path: {relative}')
    path = root / relative
    # Refuse symlinks/junctions: installation may not escape into another installation.
    for parent in [path, *path.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError(f'Linked installation path is not allowed: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes installation: {relative}')
    return path

def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']

def fetch(root, item, verify_only=False):
    path = target_path(root, item['file'])
    if valid(path, item):
        print('VERIFIED', item['file'], flush=True)
        return
    if verify_only or item.get('bundled'):
        raise RuntimeError(f"Missing or damaged asset: {item['file']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = target_path(root, item['file'] + '.part')
    print(f"DOWNLOADING {item['file']} ({item['bytes'] / 1024**2:.1f} MiB)", flush=True)
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset >= item['bytes']:
                if valid(part, item):
                    os.replace(part, path)
                    return
                part.unlink()
                offset = 0
            headers = {'User-Agent': 'Passage-Installer/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(item['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected download range')
                    mode = 'ab'
                else:
                    mode = 'wb'
                with part.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if not valid(part, item):
                part.unlink(missing_ok=True)
                raise RuntimeError('SHA-256 or size mismatch; upstream artifact may have changed')
            os.replace(part, path)
            print('INSTALLED', item['file'], flush=True)
            return
        except Exception as exc:
            print(f"RETRY {attempt + 1}/3 {item['file']}: {exc}", flush=True)
            if attempt == 2:
                raise RuntimeError(f"Failed to install {item['file']}") from exc
            time.sleep(2)

def extract_runtime(root, lock, verify_only=False):
    members = lock['runtime_members']
    if verify_only:
        for item in members:
            fetch(root, dict(item, bundled=True), True)
        return
    with zipfile.ZipFile(target_path(root, 'runtime/llama.zip')) as archive:
        for item in members:
            path = target_path(root, item['file'])
            if valid(path, item):
                continue
            part = target_path(root, item['file'] + '.part')
            with archive.open(item['member']) as source, part.open('wb') as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
            if not valid(part, item):
                part.unlink(missing_ok=True)
                raise RuntimeError(f"Runtime integrity failure: {item['file']}")
            os.replace(part, path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify all files, without network or writes')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'assets.lock.json').read_text('utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: fetch(ROOT, item, args.verify_only), lock['assets']))
    extract_runtime(ROOT, lock, args.verify_only)
    if not args.verify_only:
        path = ROOT / 'models/manifest.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(lock['assets'], indent=2) + '\n', encoding='utf-8')
        os.replace(temp, path)
        (ROOT / 'runtime/jobs').mkdir(parents=True, exist_ok=True)
    print('All assets verified.', flush=True)

if __name__ == '__main__':
    main()
