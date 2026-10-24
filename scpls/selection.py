"""Image descriptors and frozen source-fitted selection rules."""

import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .counts import nms

EPS = 1e-9
METHODS = ('DETECTOR_ONLY', 'SCORE_ONLY', 'BOX_JITTER', 'CONSISTENCY',
           'SC_PLS', 'SC_PLS_CONSISTENCY')
LABELS = dict(zip(METHODS, ('Detector-only', 'Score-only', 'BOX_JITTER',
                          'CONSISTENCY', 'SC-PLS', 'SC-PLS + consistency')))
FEATURES = ['score', 'dcs', 'H', 'Omega', 'Gamma']
PROTOCOL = {
    'inference_floor':0.05, 'candidate_gate':0.30, 'candidate_cap':60,
    'min_area_fraction':0.001, 'max_area_fraction':0.90,
    'min_aspect':0.10, 'max_aspect':10.0, 'post_selection_nms':0.50,
    'match_iou':0.50, 'high_score_cutoff':0.50,
    'jitter_count':10, 'jitter_fraction':0.06, 'jitter_seed':43,
}


def vector(raw, key, n, dtype):
    x = np.asarray(raw[key], dtype=dtype)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f'Invalid {key}: expected {n} finite values')
    return x


class ImageDCS:
    def __init__(self, raw):
        self.mean = vector(raw, 'mean', 3, np.float32)
        self.scale = vector(raw, 'std', 3, np.float32)
        self.coef = vector(raw, 'coef', 3, np.float32)
        self.bias = float(raw['bias'])
        if np.any(self.scale <= 0) or not math.isfinite(self.bias):
            raise ValueError('Invalid DCS scale or intercept')

    def score(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError('DCS requires finite [H, Omega, Gamma]')
        z = (raw - self.mean) / (self.scale + EPS)
        logit = float(np.dot(self.coef, z) + self.bias)
        with np.errstate(over='ignore'):
            return float(1.0 - 1.0 / (1.0 + np.exp(-logit)))


class BoxCalibrator:
    def __init__(self, raw, feature_names):
        if raw['features'] != feature_names:
            raise ValueError('Unexpected box feature order')
        self.features = feature_names
        n = len(feature_names)
        mean = vector(raw, 'scaler_mean', n, np.float64)
        scale = vector(raw, 'scaler_scale', n, np.float64)
        coef = vector(raw, 'coef', n, np.float64)
        bias = float(raw['intercept'])
        if np.any(scale <= 0) or not math.isfinite(bias):
            raise ValueError('Invalid box scaler or intercept')
        self.scaler = StandardScaler()
        self.scaler.mean_, self.scaler.scale_ = mean, scale
        self.scaler.var_ = scale ** 2
        self.scaler.n_features_in_, self.scaler.n_samples_seen_ = n, 1
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.model.classes_ = np.asarray([0,1],dtype=np.int64)
        self.model.coef_ = coef.reshape(1,-1)
        self.model.intercept_ = np.asarray([bias],dtype=np.float64)
        self.model.n_features_in_ = n
        self.model.n_iter_ = np.asarray([0],dtype=np.int32)

    def score(self, row):
        x = np.asarray([[row[k] for k in self.features]],dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite selection feature')
        return float(self.model.predict_proba(self.scaler.transform(x))[0,1])


def bernoulli_entropy(scores):
    if len(scores) == 0:
        return 1.0
    s = np.clip(scores, EPS, 1-EPS)
    t = 1-s
    with np.errstate(divide='ignore',invalid='ignore'):
        a, b = s*np.log(s), t*np.log(t)
    a = np.where(s == 0, 0, a)
    b = np.where(t == 0, 0, b)
    return float((-(a+b)).mean())


def descriptors(boxes, scores, width, height):
    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i+1,len(boxes)):
            a,b = boxes[i],boxes[j]
            w = max(0,min(a[2],b[2])-max(a[0],b[0]))
            h = max(0,min(a[3],b[3])-max(a[1],b[1]))
            intersection = w*h
            union = areas[i]+areas[j]-intersection
            overlaps.append(intersection/union if union>0 else 0.0)
    return np.asarray([
        bernoulli_entropy(scores),
        float(np.mean(overlaps)) if overlaps else 0.0,
        min(float(areas.sum())/max(float(width*height),1.0),1.0) if len(boxes) else 0.0,
    ],dtype=np.float32)


def candidates(boxes, scores, width, height, image_id, raw, dcs):
    rows, xyxy = [], []
    for j in np.argsort(-scores)[:PROTOCOL['candidate_cap']]:
        box = boxes[j]
        s = float(scores[j])
        x1,y1,x2,y2 = [float(v) for v in box]
        area = max(0.0,x2-x1)*max(0.0,y2-y1)/max(1.0,width*height)
        aspect = max(1e-6,x2-x1)/max(1e-6,y2-y1)
        if s < .30 or not (.001 <= area <= .90 and .10 <= aspect <= 10.):
            continue
        rows.append({'image_id':image_id,'bbox':[x1,y1,x2-x1,y2-y1],
                     'score':s,'dcs':dcs,'H':float(raw[0]),
                     'Omega':float(raw[1]),'Gamma':float(raw[2])})
        xyxy.append(box)
    return rows, np.asarray(xyxy,dtype=np.float32).reshape(-1,4)


class Selectors:
    def __init__(self, raw):
        if raw.get('schema') != 'scpls_frcnn_primary_v1':
            raise ValueError('Unsupported selector asset')
        self.thresholds = raw['thresholds']
        self.base = BoxCalibrator(raw['SC_PLS'],FEATURES)
        self.cons = BoxCalibrator(raw['SC_PLS_CONSISTENCY'],FEATURES+['cons'])
        self.cs = raw['consistency_standardization']
        self.js = raw['jitter_standardization']
        for stats in [self.cs,self.js]:
            if stats['std'] <= 0 or not all(math.isfinite(float(x)) for x in stats.values()):
                raise ValueError('Invalid stability standardization')

    def select(self, rows, methods):
        result = {m:[] for m in methods}
        for original in rows:
            r = dict(original)
            if 'cons_raw' in r:
                r['cons'] = float((r['cons_raw']-self.cs['mean'])/self.cs['std'])
            for method in methods:
                if method == 'DETECTOR_ONLY':
                    score = r['score']
                elif method == 'SCORE_ONLY':
                    score = r['score']
                elif method == 'BOX_JITTER':
                    score = float(-(r['jit_var']-self.js['mean'])/self.js['std'])
                elif method == 'CONSISTENCY':
                    score = r['cons']
                elif method == 'SC_PLS':
                    score = self.base.score(r)
                elif method == 'SC_PLS_CONSISTENCY':
                    score = self.cons.score(r)
                else:
                    raise ValueError(f'Unknown method: {method}')
                tau = self.thresholds[method]
                if tau is None or score >= tau:
                    result[method].append({'image_id':r['image_id'],'bbox':r['bbox'],
                                           'score':r['score'],'selection_score':score})
        return {m:nms(a,.50) for m,a in result.items()}
