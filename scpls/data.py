"""COCO input validation and dataset identity checks."""

import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def basename(name):
    return str(name).replace('\\','/').rsplit('/',1)[-1]


def prepare_annotations(raw):
    images=[]
    ids=set()
    for im in raw['images']:
        iid=im['id']
        if type(iid) is not int or iid in ids:
            raise ValueError('Image IDs must be distinct integers')
        ids.add(iid)
        w,h=im['width'],im['height']
        if type(w) is not int or type(h) is not int or min(w,h)<=0:
            raise ValueError(f'Invalid image dimensions: {iid}')
        if not str(im['file_name']).strip():
            raise ValueError(f'Empty file name: {iid}')
        images.append({'id':iid,'file_name':str(im['file_name']),'width':w,'height':h})
    if not images:
        raise ValueError('No images in the annotation file')
    annotations=[]
    gt={iid:[] for iid in ids}
    for index,ann in enumerate(raw['annotations'],1):
        iid=ann['image_id']
        b=[float(v) for v in ann['bbox']]
        if iid not in ids or len(b)!=4 or not all(math.isfinite(v) for v in b) or min(b[2:])<=0:
            raise ValueError(f'Invalid annotation: {index}')
        if ann.get('iscrowd',0) or ann.get('ignore',0):
            raise ValueError('Crowd/ignored annotations are outside this evaluation protocol')
        area=float(ann.get('area',b[2]*b[3]))
        if not math.isfinite(area) or area<=0:
            raise ValueError(f'Invalid annotation area: {index}')
        annotations.append({'id':index,'image_id':iid,'category_id':1,'bbox':b,'area':area,'iscrowd':0})
        gt[iid].append({'bbox':b})
    coco={'info':{},'images':images,'annotations':annotations,'categories':[{'id':1,'name':'fish'}]}
    identity={
        'images':[[x['id'],basename(x['file_name']),x['width'],x['height']] for x in images],
        'annotations':[[x['image_id'],x['bbox'],x['area']] for x in annotations],
    }
    info={'images':len(images),'fish_boxes':len(annotations),'empty_images':sum(not v for v in gt.values()),
          'semantic_sha256':json_sha256(identity)}
    return coco,gt,info


def resolve_images(root,images):
    root=Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {root}')
    paths={}
    missing=[]
    for im in images:
        name=im['file_name'].replace('\\','/')
        choices=[(root/name).resolve(),(root/basename(name)).resolve()]
        path=next((p for p in choices if p.is_relative_to(root) and p.is_file()),None)
        if path is None:
            missing.append(name)
        else:
            paths[im['id']]=path
    if missing:
        raise FileNotFoundError(f'{len(missing)} images are missing; first entries: {missing[:5]}')
    return paths


def verify_assets(repo):
    manifest=read_json(repo/'assets/SHA256SUMS.json')
    for rel,expected in manifest.items():
        path=repo/rel
        if not path.is_file() or file_sha256(path)!=expected:
            raise ValueError(f'Asset is missing or changed: {rel}. Check Git LFS if the checkpoint is a pointer.')
    return manifest
