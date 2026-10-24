"""Primary point metrics and dataset-level COCO bounding-box AP."""

import contextlib
import copy
import csv
import io
import math
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .counts import counts_for,metrics_from_counts
from .data import json_sha256
from .selection import LABELS


def normalized_rows(rows,valid_ids):
    result=[]
    for r in rows:
        iid=r['image_id']
        b=[float(v) for v in r['bbox']]
        s=float(r['score'])
        if type(iid) is not int or iid not in valid_ids:
            raise ValueError('Prediction has an unknown image ID')
        if len(b)!=4 or not all(math.isfinite(v) for v in b) or min(b[2:])<=0:
            raise ValueError('Invalid prediction box')
        if not math.isfinite(s) or not 0<=s<=1:
            raise ValueError('Invalid detector score')
        result.append({'image_id':iid,'bbox':b,'score':s})
    return result


def row_sha256(rows):
    return json_sha256([[r['image_id'],r['bbox'],r['score']] for r in rows])


def coco_metrics(coco,rows,ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO()
        gt.dataset=copy.deepcopy(coco)
        gt.createIndex()
        predictions=[dict(r,category_id=1) for r in rows]
        if predictions:
            dt=gt.loadRes(predictions)
        else:
            dt=COCO()
            dt.dataset={'images':copy.deepcopy(coco['images']),
                        'categories':copy.deepcopy(coco['categories']),'annotations':[]}
            dt.createIndex()
        ev=COCOeval(gt,dt,'bbox')
        ev.params.imgIds=list(ids)
        ev.params.catIds=[1]
        ev.params.useCats=1
        ev.params.maxDets=[1,10,100]
        if not np.array_equal(ev.params.iouThrs,np.linspace(.5,.95,10)):
            raise ValueError('Unexpected COCO IoU grid')
        if not np.array_equal(ev.params.recThrs,np.linspace(0,1,101)):
            raise ValueError('Unexpected COCO recall grid')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {'COCO_AP50':float(ev.stats[1]),'COCO_AP50_95':float(ev.stats[0])}


def evaluate(coco,gt,rows):
    ids=sorted(gt)
    clean=normalized_rows(rows,set(ids))
    per_image=counts_for(clean,gt,ids)
    result=metrics_from_counts(per_image,ids)
    result['accepted_boxes']=len(clean)
    result.update(coco_metrics(coco,clean,ids))
    result['ordered_rows_sha256']=row_sha256(clean)
    return result,per_image


def compare_reference(results,reference):
    checks={}
    keys_int=['tp','fp','fn','hs_fp','accepted_boxes']
    keys_rate=['precision','recall','f1','COCO_AP50','COCO_AP50_95']
    for method,actual in results.items():
        expected=reference['methods'][method]
        different={}
        for key in keys_int+keys_rate:
            a,b=actual[key],expected[key]
            equal=(a==b) if key in keys_int else (f'{a:.4f}'==f'{b:.4f}')
            if not equal:
                different[key]={'actual':a,'reference':b,'difference':a-b}
        checks[method]={
            'table_match':not different,
            'matches_unrounded_metrics':all(actual[k]==expected[k] for k in keys_int) and
                all(math.isclose(actual[k],expected[k],rel_tol=0,abs_tol=1e-12) for k in keys_rate),
            'ordered_detection_values_match':actual['ordered_rows_sha256']==expected['ordered_rows_sha256'],
            'metric_differences':{k:actual[k]-expected[k] for k in keys_int+keys_rate},
            'table_mismatches':different,
        }
    passed=bool(checks) and all(v['table_match'] for v in checks.values())
    return {'status':'TABLE_MATCH' if passed else 'DIFFERENCES_FOUND',
            'scope':'Requested methods; exact counts and rates rounded to four decimal places.',
            'methods':checks}


def write_table(path,results,thresholds):
    fields=['Method','tau','P','R','F1','Accepted boxes','HS-FP','COCO AP50','COCO AP50:95']
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(fields)
        for m,r in results.items():
            t=thresholds[m]
            writer.writerow([LABELS[m],'' if t is None else f'{t:.4f}',
                             f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}",
                             r['accepted_boxes'],r['hs_fp'],f"{r['COCO_AP50']:.4f}",f"{r['COCO_AP50_95']:.4f}"])
