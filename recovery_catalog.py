"""Inventory retained OCR without changing any comic. Optional text-only retranslation."""
import argparse,json,re
from pathlib import Path
import engine

def record_paths(job):
    """Primary records win; recovered text snapshots never replace them."""
    files={str(p.relative_to(job/'working')):p for p in (job/'working').rglob('*.json')}
    recovered=engine.ROOT/'runtime/recovered'/job.name
    for p in recovered.rglob('*.json'):files.setdefault(str(p.relative_to(recovered)),p)
    return list(sorted(files.items()))

def recover_evidence_records():
    """Recover the lab corpus's explicitly named jobs, without touching any comic."""
    corpus=engine.ROOT.parent/'Passage-lab/out/corpus.json'
    if not corpus.is_file():return []
    recovered=[]
    for page in json.loads(corpus.read_text(encoding='utf-8')):
        prefix=page.get('book','').removeprefix('job-')
        if not re.fullmatch('[a-f0-9]{4,24}',prefix):continue
        jobs=[p for p in (engine.ROOT/'runtime/jobs').glob(prefix+'*') if (p/'manifest.json').is_file()]
        if len(jobs)!=1:continue
        job=jobs[0];name=page['page']
        if Path(name).name!=name:continue
        target=engine.ROOT/'runtime/recovered'/job.name/(name+'.json')
        if (job/'working'/(name+'.json')).exists() or target.exists():continue
        original=engine.ROOT/'evidence/floating-fix-20261106'/(name+'.json')
        rows=[]
        if original.exists():
            existing=json.loads(original.read_text(encoding='utf-8')).get('regions',[])
            expected={r['id']:r['source'] for r in page['items']}
            rows=[r for r in existing if expected.get(r['id'])==r.get('source')]
        if not rows:
            rows=[{'id':r['id'],'source':r['source'],'translation':r.get('old',''),
                   'kind':r.get('old_kind','unknown'),'status':r.get('old_status','历史恢复')} for r in page['items']]
        engine.atomic_json(target,{'regions':rows,'note':'从实验文字记录恢复；禁止在缺少原图时直接重新嵌字。','recovered_from':str(corpus),'page':name})
        recovered.append(str(target))
    return recovered

def catalog():
    jobs=[]
    for job in sorted((engine.ROOT/'runtime/jobs').iterdir()):
        manifest=job/'manifest.json'
        if not manifest.is_file():continue
        data=json.loads(manifest.read_text(encoding='utf-8'));pages=[]
        for relative,meta in record_paths(job):
            try:record=json.loads(meta.read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            regions=record.get('regions',[])
            ja=[r for r in regions if re.search('[\u3040-\u30ff]',r.get('source',''))]
            source_regions=[r for r in regions if r.get('source')]
            if not source_regions:continue
            rel=relative[:-5]
            pages.append({'page':rel,'metadata':str(meta),'japanese_regions':len(ja),
                'source_regions':len(source_regions),
                'with_layout':sum(bool(r.get('layout')) for r in ja),
                'source_exists':(Path(data['source'])/rel).is_file(),
                'output_at_recorded_path':(Path(data['output'])/rel).is_file()})
        jobs.append({'job':job.name,'source':data.get('source'),'output':data.get('output'),'language':'ja' if any(p['japanese_regions'] for p in pages) else data.get('language','unknown'),
                     'manifest_pages':len(data.get('pages',{})),'recoverable_pages':pages})
    return {'jobs':jobs,'note':'OCR文字是识别结果，不保证与原文完全一致。路径不存在不等于文件不存在，可能已搬移。'}

def retranslate(job_id,destination):
    from task_storage import JobLock
    if not re.fullmatch('[a-f0-9]{24}',job_id):raise ValueError('Invalid job id')
    job=engine.ROOT/'runtime/jobs'/job_id;dest=Path(destination).resolve()
    manifest=json.loads((job/'manifest.json').read_text(encoding='utf-8'))
    for protected in (Path(manifest['source']).resolve(),Path(manifest['output']).resolve(),engine.ROOT/'runtime/jobs'):
        if dest==protected or protected in dest.parents:raise ValueError('请选择漫画与内部记录以外的目录保存文字对照。')
    target=dest/f'{job_id}-sakura-text.json'
    if target.exists():raise FileExistsError('对比文件已存在，请选择新目录。')
    if not __import__('sakura').available():raise FileNotFoundError('Sakura 模型未就绪。')
    dest.mkdir(parents=True,exist_ok=True)
    engine.offline_guard();translator=engine.LocalTranslator(print);reports=[]
    lock=JobLock(dest);lock.__enter__()
    try:
        if target.exists():raise FileExistsError('对比文件已存在，请选择新目录。')
        for relative,meta in record_paths(job):
            data=json.loads(meta.read_text(encoding='utf-8'))
            rows=[dict(r) for r in data.get('regions',[]) if r.get('source') and r.get('kind') not in ('sfx','preserve')]
            old={r['id']:r.get('translation','') for r in rows}
            for r in rows:r.update(translation='',status='待翻译')
            for start in range(0,len(rows),4):translator.translate(rows[start:start+4],'ja','',[])
            reports.append({'page':relative,'items':[
                {'id':r['id'],'source':r['source'],'old':old[r['id']],'new':r.get('translation',''),
                 'status':r.get('status'),'box':r.get('box'),'old_layout':r.get('layout')} for r in rows]})
        engine.atomic_json(target,reports)
    finally:translator.close();lock.__exit__(None,None,None)
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retranslate');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.retranslate:print(retranslate(args.retranslate,args.output))
    else:engine.atomic_json(Path(args.output),catalog())
