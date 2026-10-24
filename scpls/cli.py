"""Commands for preflight checks, primary reproduction, and annotation."""

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

from PIL import Image
from tqdm import tqdm

from .data import read_json,write_json,file_sha256,prepare_annotations,resolve_images,verify_assets
from .selection import METHODS,LABELS,PROTOCOL
from .runtime import configure_numeric_runtime
from . import __version__

REPO=Path(__file__).resolve().parent.parent


def method_list(name):
    if name=='all':
        return METHODS
    if name=='core':
        return ('DETECTOR_ONLY','SCORE_ONLY','SC_PLS')
    if name in METHODS:
        return (name,)
    raise ValueError(f'Invalid methods value: {name}')


def environment():
    numeric_runtime=configure_numeric_runtime()
    names=['torch','torchvision','numpy','Pillow','scikit-learn','scipy','pycocotools']
    versions={n:importlib.metadata.version(n) for n in names}
    import torch
    return {'release_version':__version__,'python':sys.version,'platform':platform.platform(),'packages':versions,
            'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'numeric_runtime':numeric_runtime}


def load_inputs(config):
    path=Path(config['annotations']).expanduser().resolve()
    coco,gt,identity=prepare_annotations(read_json(path))
    reference=None
    if config['dataset']!='custom':
        reference=read_json(REPO/'assets/reference_results.json')[config['dataset']]
        expected=reference['dataset']
        for key in ['images','fish_boxes','empty_images','semantic_sha256']:
            if identity[key]!=expected[key]:
                raise ValueError(f"Dataset does not match the paper ({key}): {identity[key]} != {expected[key]}. "
                                 'Use the correct annotation version; use --dataset custom only for a separate evaluation.')
    paths=resolve_images(config['images'],coco['images'])
    identity['annotation_file_sha256']=file_sha256(path)
    return coco,gt,identity,paths,reference


def make_output(path):
    out=Path(path).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}. Choose a new --output directory.')
    out.mkdir(parents=True)
    return out


def load_engine(config):
    from .inference import configure_device,load_detector,Engine
    device=configure_device(config['device'])
    model=load_detector(REPO/'assets/detector_best.pth',device)
    return Engine(model,device,read_json(REPO/'assets/dcs_calibrator.json'),
                  read_json(REPO/'assets/selectors.json'),method_list(config['methods']))


def make_review_zip(out):
    with zipfile.ZipFile(out/'review.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='review.zip':
                z.write(p,p.relative_to(out).as_posix())


def reproduce(config):
    from .evaluation import evaluate,compare_reference,write_table
    assets=verify_assets(REPO)
    coco,gt,identity,paths,reference=load_inputs(config)
    out=make_output(config['output'])
    write_json(out/'PREFLIGHT.json',{'status':'PASS','dataset':identity,'assets':assets})
    started=time.perf_counter()
    images=coco['images']
    limit=config.get('max_images',0)
    partial=bool(limit and limit<len(images))
    if partial:
        images=images[:limit]
        ids={x['id'] for x in images}
        coco=dict(coco,images=images,annotations=[a for a in coco['annotations'] if a['image_id'] in ids])
        gt={k:v for k,v in gt.items() if k in ids}
    info={'dataset':config['dataset'],'partial_run':partial,'images':len(images),
          'methods':list(method_list(config['methods'])),'protocol':PROTOCOL,
          'environment':environment(),'device':config['device'],
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'annotations_used_for_selection':False,'parameters_refitted':False}
    write_json(out/'RUN.json',info)
    try:
        engine=load_engine(config)
        import torch
        info['device_name']=torch.cuda.get_device_name(engine.device) if engine.device.type=='cuda' else 'CPU'
        predictions={m:[] for m in engine.methods}
        image_hashes=[]
        candidate_count=0
        for im in tqdm(images,desc='Faster R-CNN',unit='image'):
            p=paths[im['id']]
            digest=file_sha256(p)
            with Image.open(p) as original:
                image=original.convert('RGB')
            if image.size!=(im['width'],im['height']):
                raise ValueError(f"Image dimensions differ from annotations: {im['file_name']}")
            selected,n=engine.predict_image(image,im['id'])
            candidate_count+=n
            for method,rows in selected.items():
                predictions[method].extend(rows)
            image_hashes.append({'image_id':im['id'],'file_name':im['file_name'],'sha256':digest})
        (out/'predictions').mkdir()
        for method,rows in predictions.items():
            write_json(out/'predictions'/f'{method}.json',
                       [dict(image_id=r['image_id'],category_id=1,bbox=r['bbox'],score=r['score']) for r in rows])
        write_json(out/'IMAGE_HASHES.json',image_hashes)
        results,per_image={},{}
        for method,rows in predictions.items():
            print(f'Evaluating {LABELS[method]}...',flush=True)
            results[method],per_image[method]=evaluate(coco,gt,rows)
        write_json(out/'RESULTS.json',results)
        write_json(out/'PER_IMAGE_COUNTS.json',{'columns':['TP','FP','FN','HS-FP'],'methods':per_image})
        write_table(out/'table.csv',results,engine.selectors.thresholds)
        if reference is not None and not partial:
            comparison=compare_reference(results,reference)
        else:
            comparison={'status':'SMOKE_COMPLETE' if partial else 'CUSTOM_EVALUATION_COMPLETE',
                        'paper_comparison_performed':False}
        comparison['methods_evaluated']=list(results)
        comparison['complete_six_method_table']=reference is not None and not partial and tuple(results)==METHODS
        write_json(out/'COMPARISON.json',comparison)
        info.update(status=comparison['status'],candidate_boxes=candidate_count,
                    elapsed_seconds=time.perf_counter()-started)
        write_json(out/'RUN.json',info)
        print((out/'table.csv').read_text(encoding='utf-8'))
        print(f"Status: {comparison['status']}\nOutput: {out}\nReview archive: {out/'review.zip'}")
        make_review_zip(out)
        return 2 if comparison['status']=='DIFFERENCES_FOUND' else 0
    except Exception as exc:
        info.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'RUN.json',info)
        raise


def predict(config):
    verify_assets(REPO)
    root=Path(config['images']).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    extensions={'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'}
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise ValueError('No supported images found')
    out_path=Path(config['output']).expanduser().resolve()
    if out_path.is_relative_to(root):
        raise ValueError('Choose an output directory outside the input image directory')
    out=make_output(out_path)
    engine=load_engine(config)
    methods=engine.methods
    images=[]
    rows={m:[] for m in methods}
    for iid,path in enumerate(tqdm(paths,desc='Selecting',unit='image')):
        with Image.open(path) as original:
            image=original.convert('RGB')
        images.append({'id':iid,'file_name':path.relative_to(root).as_posix(),
                       'width':image.width,'height':image.height})
        selected,_=engine.predict_image(image,iid)
        for m,a in selected.items():
            rows[m].extend(a)
    for m,a in rows.items():
        anns=[{'id':i,'image_id':r['image_id'],'category_id':1,'bbox':r['bbox'],
               'area':r['bbox'][2]*r['bbox'][3],'iscrowd':0,'score':r['score']}
              for i,r in enumerate(a,1)]
        write_json(out/f'{m}_pseudo_labels.json',{'images':images,'annotations':anns,
                                               'categories':[{'id':1,'name':'fish'}]})
    write_json(out/'RUN.json',{'status':'PREDICTION_COMPLETE','methods':list(methods),
                             'protocol':PROTOCOL,'environment':environment(),
                             'target_labels_used':False,'parameters_refitted':False})
    print(f'Selected detections: {out}')
    return 0


def main(argv=None):
    configure_numeric_runtime()
    p=argparse.ArgumentParser(description='SC-PLS frozen Faster R-CNN primary evaluation')
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify',help='Check assets, dependencies, and optionally load the checkpoint')
    v.add_argument('--load-model',action='store_true')
    v.add_argument('--device',default='cpu')
    for command in ['preflight','reproduce']:
        q=sub.add_parser(command)
        q.add_argument('--config',type=Path)
        q.add_argument('--images')
        q.add_argument('--annotations')
        q.add_argument('--dataset',choices=['tefimo','sustaintrawl','custom'])
        q.add_argument('--device')
        q.add_argument('--output')
        q.add_argument('--methods',choices=['all','core']+list(METHODS))
        q.add_argument('--max-images',type=int)
    q=sub.add_parser('predict',help='Select pseudo-labels from unlabelled images')
    q.add_argument('--images',required=True)
    q.add_argument('--output',required=True)
    q.add_argument('--device',default='cuda:0')
    q.add_argument('--methods',choices=['all','core']+list(METHODS),default='SC_PLS')
    args=vars(p.parse_args(argv))
    command=args.pop('command')
    try:
        if command=='verify':
            verified=verify_assets(REPO)
            print(json.dumps(environment(),indent=2))
            print(f'Asset SHA256 checks: {len(verified)} passed')
            if args['load_model']:
                from .inference import configure_device,load_detector
                model=load_detector(REPO/'assets/detector_best.pth',configure_device(args['device']))
                print(f'Checkpoint loaded strictly: {sum(p.numel() for p in model.parameters()):,} parameters')
            return 0
        config={'device':'cuda:0','methods':'all','max_images':0}
        path=args.pop('config',None)
        if path:
            config.update(read_json(path))
        config.update({k:v for k,v in args.items() if v is not None})
        if command=='predict':
            return predict(config)
        for key in ['images','annotations','dataset']:
            if not config.get(key):
                raise ValueError(f'Missing --{key} or config field')
        if config['dataset'] not in ['tefimo','sustaintrawl','custom']:
            raise ValueError('Unknown dataset')
        if config['max_images']<0:
            raise ValueError('--max-images must be nonnegative')
        method_list(config['methods'])
        if command=='preflight':
            verify_assets(REPO)
            _,_,identity,_,_=load_inputs(config)
            print(json.dumps({'status':'PASS','dataset':identity},indent=2))
            return 0
        if not config.get('output'):
            raise ValueError('Missing --output or config field')
        return reproduce(config)
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
