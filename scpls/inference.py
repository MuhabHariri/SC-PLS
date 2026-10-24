"""Target-only inference with the frozen Faster R-CNN checkpoint."""

import random
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from .selection import ImageDCS, Selectors, descriptors, candidates, PROTOCOL
from .stability import jitter_variance_for_boxes, _iou_matrix
from .runtime import configure_numeric_runtime


def configure_device(name):
    configure_numeric_runtime()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    device = torch.device(name)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install the CUDA requirements or use --device cpu.')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    return device


def load_detector(path, device):
    model = fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None)
    n = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(n,2)
    checkpoint = torch.load(str(path),map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def detect(model, tensor, device):
    out = model([tensor.to(device)])[0]
    mask = out['scores'] >= PROTOCOL['inference_floor']
    return (out['boxes'][mask].cpu().numpy(), out['scores'][mask].cpu().numpy())


@torch.no_grad()
def flip_consistency(model, image, device, boxes, scores):
    out = np.zeros(len(boxes),np.float32)
    if not len(boxes):
        return out
    fb, fs = detect(model,TF.to_tensor(TF.hflip(image)),device)
    unflipped = fb.copy()
    if len(fb):
        unflipped[:,0] = image.width-fb[:,2]
        unflipped[:,2] = image.width-fb[:,0]
    iou = _iou_matrix(boxes,unflipped)
    for i in range(len(boxes)):
        if not iou.shape[1]:
            continue
        j = int(np.argmax(iou[i]))
        if iou[i,j] >= .50:
            out[i] = float(max(0.0,min(1.0,1.0-abs(float(scores[i])-float(fs[j])))))
    return out


class Engine:
    def __init__(self, model, device, dcs_asset, selector_asset, methods):
        self.model, self.device = model, device
        self.dcs, self.selectors = ImageDCS(dcs_asset), Selectors(selector_asset)
        self.methods = tuple(methods)
        self.rng = np.random.default_rng(PROTOCOL['jitter_seed'])

    @torch.no_grad()
    def predict_image(self, image, image_id):
        tensor = TF.to_tensor(image)
        boxes,scores = detect(self.model,tensor,self.device)
        raw = descriptors(boxes,scores,image.width,image.height)
        dcs = self.dcs.score(raw)
        rows,cb = candidates(boxes,scores,image.width,image.height,image_id,raw,dcs)
        if rows and any(m in self.methods for m in ['CONSISTENCY','SC_PLS_CONSISTENCY']):
            cs = np.asarray([r['score'] for r in rows],np.float32)
            consistency = flip_consistency(self.model,image,self.device,cb,cs)
            for r,v in zip(rows,consistency):
                r['cons_raw'] = float(v)
        if rows and 'BOX_JITTER' in self.methods:
            dispersion = jitter_variance_for_boxes(self.model,tensor,cb,10,.06,self.rng)
            for r,v in zip(rows,dispersion):
                r['jit_var'] = float(v)
        return self.selectors.select(rows,self.methods), len(rows)
