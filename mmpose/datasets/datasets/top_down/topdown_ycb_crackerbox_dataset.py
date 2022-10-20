# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownYCBCrackerBoxDataset(Kpt2dSviewRgbImgTopDownDataset):
    """
    Dataset to train ViTPose on CrackerBox from YCB 

        0: 'cuboid_keypoint'
        1: 'cuboid_keypoint'
        2: 'cuboid_keypoint'
        3: 'cuboid_keypoint'
        4: 'cuboid_keypoint'
        5: 'cuboid_keypoint'
        6: 'cuboid_keypoint'
        7: 'cuboid_keypoint'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        print("Initializing TopDownYCBCrackerBoxDataset...")

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=False)

        self.ann_file = ann_file 
        self.bbox_count = 0

        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)


        self.width = self.ann_info["image_size"][0]
        self.height = self.ann_info["image_size"][1]
        # self.vis_thr = data_cfg['vis_thr']
        
        self.db = self._get_db()

        print(f'=> load {len(self.db)} samples')


    def _get_db(self):
        """Load dataset."""
        gt_db = self._load_keypoint_annotations()

        return gt_db


    def _load_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []

        bbox_count, prev_image_id = 0, 0 
        with open(self.ann_file) as f:
            annotations = json.load(f)

            for i, obj in enumerate(annotations):
                self.bbox_count += 1 
                num_joints = self.ann_info['num_joints']
                image_file = osp.join(self.img_prefix, obj["image_id"].lstrip("/"))

                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 2)
                joints_3d[:, :2] = keypoints[:, :2]

                for i, pair in enumerate(keypoints):
                    if pair[0] < 0 or pair[1] < 0 or pair[0] > self.width or pair[1] > self.height:
                        joints_3d_visible[i, :2] = 0
                    else:
                        joints_3d_visible[i, :2] = 1

                gt_db.append({
                    'image_file': image_file,
                    'center': np.array([obj['bbox'][0] + obj['bbox'][2] / 2.0, obj['bbox'][1] + obj['bbox'][3] / 2.0], dtype=np.float32),
                    'scale': np.array([1.0, 1.0], dtype=np.float32),
                    'bbox': obj['bbox'],
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox_score': 1,
                    'bbox_id': bbox_count  
                })
                
                if prev_image_id == obj["image_id"]:
                    bbox_count += 1 
                else:
                    prev_image_id = obj["image_id"]
                    bbox_count = 0 

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            raise Exception("res_folder cannot be None")

        detections = defaultdict(dict)

        # kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            # boxes = result['boxes']
            image_paths = result['image_paths']
            # bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                key = image_paths[i]

                keypoints = preds[i]

                projected_cuboid = [[point[0], point[1]] for point in keypoints]

                if "objects" not in detections[key]:
                    detections[key]["objects"] = []
                    detections[key]["full_file_path"] = osp.join(os.getcwd(), key)

                detections[key]["objects"].append({
                    "projected_cuboid": projected_cuboid
                })
                
        with open(res_file, 'w') as f:
            json.dump(detections, f, indent=4)

        return {}
