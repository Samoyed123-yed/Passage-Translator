"""Cross-process output locks and recoverable page commits. No image reads by default."""
import contextlib,functools,hashlib,json,os,threading,uuid
from pathlib import Path

_owners={};_guard=threading.RLock()

class JobLock:
    def __init__(self,output):
        key=hashlib.sha256(str(Path(output).resolve()).casefold().encode()).hexdigest()
        self.path=Path(__file__).parent/'runtime/locks'/f'{key}.lock'
        self.owner=threading.get_ident();self.nested=False
    def __enter__(self):
        with _guard:
            if self.path in _owners:
                if _owners[self.path][0]!=self.owner:raise RuntimeError('此输出目录已有任务运行，请先停止原任务。')
                _owners[self.path][1]+=1;self.nested=True;return self
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.file=self.path.open('a+b');self.file.seek(0)
            if self.path.stat().st_size==0:self.file.write(b'0');self.file.flush()
            self.file.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                self.file.close();raise RuntimeError('此输出目录已有其他进程正在写入。')
            _owners[self.path]=[self.owner,1,self.file]
        return self
    def __exit__(self,*args):
        with _guard:
            entry=_owners[self.path];entry[1]-=1
            if entry[1]:return
            f=entry[2];f.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)
            f.close();del _owners[self.path]

def locked_run(method):
    @functools.wraps(method)
    def run(self,*args,**kwargs):
        with JobLock(self.output):
            from automatic import read_manifest
            self.state=read_manifest(self.source,self.output) or self.state
            recover_commits(self)
            return method(self,*args,**kwargs)
    return run

def commit_page(batch,key,dst,write_image,record,metadata):
    import engine
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    stage=dst.with_name('.passage-stage-'+uuid.uuid4().hex+dst.suffix)
    pending=batch.job/'pending'/f'{uuid.uuid4().hex}.json'
    try:
        write_image(stage)
        record=dict(record,output_sha256=engine.sha256(stage))
        metadata=dict(metadata,output_sha256=record['output_sha256'])
        payload={'key':key,'stage':str(stage),'target':str(dst),'record':record,'metadata':metadata,
                 'previous_hash':engine.sha256(dst) if dst.exists() else None}
        engine.atomic_json(pending,payload)  # durable intent before replacing the page
    except BaseException:
        # No intent was recorded, so nothing can recover this stage: never leave a
        # hidden .passage-stage file in the reader's folder.
        if not pending.exists() and stage.exists():stage.unlink()
        raise
    _finish(batch,payload,pending)
    return record

def _finish(batch,payload,pending):
    import engine
    from automatic import checked_path
    dst=checked_path(batch.output,Path(payload['target']))
    stage=checked_path(batch.output,Path(payload['stage']))
    key=payload['key'];record=payload['record']
    source=checked_path(batch.source,batch.source/key)
    if engine.sha256(source)!=record['source_sha256']:raise RuntimeError('恢复写入时发现原图已变化，已停止。')
    current=engine.sha256(dst) if dst.exists() else None
    if current!=record['output_sha256']:
        if current!=payload['previous_hash']:raise RuntimeError('输出已被外部修改，未覆盖。')
        if not stage.is_file() or engine.sha256(stage)!=record['output_sha256']:raise RuntimeError('暂存图片校验失败。')
        stage.replace(dst)
    elif stage.exists():stage.unlink()
    meta=checked_path(batch.job,batch.job/'working'/Path(key).parent/(Path(key).name+'.json'))
    engine.atomic_json(meta,payload['metadata'])
    batch.state.setdefault('pages',{})[key]=record
    engine.atomic_json(batch.job/'manifest.json',batch.state)
    pending.unlink()

def recover_commits(batch):
    for pending in sorted((batch.job/'pending').glob('*.json')):
        _finish(batch,json.loads(pending.read_text(encoding='utf-8')),pending)

def rerender_page(meta_path,draft=None):
    import engine
    from automatic import AutoBatch,save_matching,checked_path,read_manifest,unresolved_count
    from PIL import Image
    meta=Path(meta_path).resolve();saved=json.loads(meta.read_text(encoding='utf-8'))
    # Geometry and paths remain trusted saved state; UI edits only translation fields.
    record=json.loads(json.dumps(saved))
    if draft is not None:
        edits={r['id']:r for r in draft['regions']}
        for r in record['regions']:
            for field in ('translation','kind','direction_override','status'):
                if field in edits.get(r['id'],{}):r[field]=edits[r['id']][field]
            r.pop('skip_review',None);r.pop('automatic_reason',None)
    if not record.get('job'):raise ValueError('旧记录缺少任务路径，请在历史记录中导出文字；重新嵌字需要原图和完整记录。')
    job=Path(record['job']).resolve();manifest=json.loads((job/'manifest.json').read_text(encoding='utf-8'))
    with JobLock(manifest['output']):
        batch=AutoBatch(manifest['source'],manifest['output'],record['language'],log=lambda _:None)
        if batch.job.resolve()!=job or meta!=checked_path(job,job/'working'/Path(record['key']).parent/(Path(record['key']).name+'.json')):raise ValueError('校对记录与任务不匹配。')
        recover_commits(batch)
        source=checked_path(batch.source,batch.source/record['key']);dst=checked_path(batch.output,batch.output/record['key'])
        if not source.is_file():raise ValueError('缺少原图：可查看和导出译文，但无法安全重新嵌字。')
        if engine.sha256(source)!=record['source_sha256']:raise ValueError('原图已经改变，未写入。')
        old=batch.state['pages'][record['key']]
        if engine.sha256(dst)!=old['output_sha256']:raise ValueError('成品已被修改，未覆盖。')
        engine.offline_guard()
        with Image.open(source) as im:
            image=im.convert('RGB');alpha=im.getchannel('A') if 'A' in im.getbands() else None
        _,mask=batch.vision.detect(image)
        result=engine.render(image,mask,record['regions'],batch.vision,record['language'],automatic=True)
        if alpha is not None:result.putalpha(alpha)
        unresolved=unresolved_count(record['regions']);record.update(review_count=unresolved,version=engine.VERSION)
        updated=dict(old,retained=bool(unresolved),unresolved_regions=unresolved,
                     translated_regions=sum(r.get('status')=='已嵌字' for r in record['regions']))
        committed=commit_page(batch,record['key'],dst,lambda path:save_matching(result,path),updated,record)
        record['output_sha256']=committed['output_sha256'];batch.translator.close()
        return record
