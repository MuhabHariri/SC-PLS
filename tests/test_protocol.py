import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scpls.counts import counts_for,metrics_from_counts,nms
from scpls.data import read_json,prepare_annotations,resolve_images
from scpls.evaluation import evaluate,compare_reference
from scpls.selection import ImageDCS,Selectors,bernoulli_entropy,descriptors,candidates

ROOT=Path(__file__).resolve().parents[1]


def sample_annotations():
    return {'images':[{'id':0,'file_name':'a.png','width':100,'height':100},
                      {'id':1,'file_name':'b.png','width':100,'height':100}],
            'annotations':[{'id':8,'image_id':0,'category_id':7,'bbox':[10,10,20,20]}]}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset=read_json(ROOT/'assets/selectors.json')
        cls.selectors=Selectors(cls.asset)

    def test_entropy_boundary_scores(self):
        self.assertEqual(bernoulli_entropy(np.asarray([1.0],np.float32)),0.0)
        self.assertTrue(math.isfinite(bernoulli_entropy(np.asarray([0.0,.5,1.0],np.float32))))
        self.assertEqual(bernoulli_entropy(np.asarray([],np.float32)),1.0)

    def test_entropy_retains_interior_arithmetic(self):
        rng=np.random.default_rng(4)
        for _ in range(30):
            s=rng.uniform(.05,.999,100).astype(np.float32)
            expected=float((-(s*np.log(s)+(1-s)*np.log(1-s))).mean())
            self.assertEqual(bernoulli_entropy(s),expected)

    def test_descriptor_definitions(self):
        raw=descriptors(np.asarray([[0,0,20,20],[0,0,20,20]],np.float32),
                        np.asarray([.5,.5],np.float32),100,100)
        self.assertAlmostEqual(float(raw[0]),math.log(2),places=6)
        self.assertEqual(float(raw[1]),1.0)
        self.assertAlmostEqual(float(raw[2]),.08,places=7)

    def test_candidates_use_inclusive_gate(self):
        b=np.asarray([[0,0,20,20],[30,30,50,50]],np.float32)
        rows,_=candidates(b,np.asarray([.30,.299],np.float32),100,100,4,[.2,.1,.3],.5)
        self.assertEqual(len(rows),1)

    def test_candidate_cap_precedes_geometry_filter(self):
        b=np.asarray([[0,0,1,1]]*60+[[0,0,20,20]],np.float32)
        s=np.linspace(.99,.31,61,dtype=np.float32)
        rows,_=candidates(b,s,100,100,4,[.2,.1,.3],.5)
        self.assertEqual(rows,[])

    def test_original_scores_and_boxes_preserved(self):
        r={'image_id':0,'bbox':[1.,2.,20.,20.],'score':.9,'dcs':.2,'H':.2,'Omega':.1,'Gamma':.3}
        for m,rows in self.selectors.select([r],['DETECTOR_ONLY','SCORE_ONLY','SC_PLS']).items():
            for result in rows:
                self.assertEqual(result['bbox'],r['bbox'])
                self.assertEqual(result['score'],r['score'])

    def test_score_only_threshold_inclusive(self):
        r={'image_id':0,'bbox':[0,0,10,10],'score':self.asset['thresholds']['SCORE_ONLY']}
        self.assertEqual(len(self.selectors.select([r],['SCORE_ONLY'])['SCORE_ONLY']),1)

    def test_nms_suppresses_at_exact_iou_half(self):
        rows=[{'image_id':0,'bbox':[0,0,3,1],'score':.9},
              {'image_id':0,'bbox':[1,0,3,1],'score':.8}]
        self.assertEqual(len(nms(rows,.5)),1)

    def test_empty_images_and_duplicate_detections(self):
        gt={0:[{'bbox':[0,0,10,10]}],1:[]}
        rows=[{'image_id':0,'bbox':[0,0,10,10],'score':.9},
              {'image_id':0,'bbox':[0,0,10,10],'score':.8},
              {'image_id':1,'bbox':[0,0,10,10],'score':.5}]
        r=metrics_from_counts(counts_for(rows,gt,[0,1]),[0,1])
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['hs_fp']),(1,2,0,2))

    def test_matching_uses_unmatched_ground_truth(self):
        gt={0:[{'bbox':[0,0,10,10]},{'bbox':[0,0,10,10]}]}
        rows=[{'image_id':0,'bbox':[0,0,10,10],'score':.9},
              {'image_id':0,'bbox':[0,0,10,10],'score':.8}]
        self.assertEqual(counts_for(rows,gt,[0])[0],(2,0,0,0))

    def test_class_agnostic_dataset_identity(self):
        a=sample_annotations()
        _,_,i1=prepare_annotations(a)
        a['annotations'][0]['category_id']=44
        _,_,i2=prepare_annotations(a)
        self.assertEqual(i1,i2)
        a['annotations'][0]['bbox'][0]+=1
        self.assertNotEqual(prepare_annotations(a)[2]['semantic_sha256'],i1['semantic_sha256'])

    def test_image_order_is_part_of_dataset_identity(self):
        a=sample_annotations()
        h=prepare_annotations(a)[2]['semantic_sha256']
        a['images'].reverse()
        self.assertNotEqual(h,prepare_annotations(a)[2]['semantic_sha256'])

    def test_no_silent_missing_images(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                resolve_images(d,sample_annotations()['images'])

    def test_no_silent_annotation_filtering(self):
        a=sample_annotations()
        a['annotations'][0]['iscrowd']=1
        with self.assertRaises(ValueError):
            prepare_annotations(a)

    def test_coco_ap_includes_empty_images(self):
        coco,gt,_=prepare_annotations(sample_annotations())
        rows=[{'image_id':1,'bbox':[10,10,20,20],'score':.9},
              {'image_id':0,'bbox':[10,10,20,20],'score':.8}]
        metrics,_=evaluate(coco,gt,rows)
        self.assertEqual(metrics['fp'],1)
        self.assertAlmostEqual(metrics['COCO_AP50'],.5,places=12)

    def test_empty_predictions_are_supported(self):
        coco,gt,_=prepare_annotations(sample_annotations())
        r,_=evaluate(coco,gt,[])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(0,0,1))
        self.assertEqual(r['COCO_AP50'],0.0)

    def test_comparison_does_not_hide_integer_difference(self):
        r=read_json(ROOT/'assets/reference_results.json')['tefimo']
        actual=copy.deepcopy(r['methods'])
        actual['SC_PLS']['hs_fp']+=1
        check=compare_reference(actual,r)
        self.assertEqual(check['status'],'DIFFERENCES_FOUND')
        self.assertIn('hs_fp',check['methods']['SC_PLS']['table_mismatches'])

    def test_dcs_and_box_scores_are_finite(self):
        d=ImageDCS(read_json(ROOT/'assets/dcs_calibrator.json'))
        for raw in [[0,0,0],[1,0,0],[.4,.1,.8]]:
            score=d.score(raw)
            self.assertTrue(0<=score<=1)
            row=dict(zip(['score','dcs','H','Omega','Gamma'],[.6,score,*raw]))
            self.assertTrue(0<=self.selectors.base.score(row)<=1)


if __name__=='__main__':
    unittest.main()
