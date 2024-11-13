# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PascalVOCDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'door', 'other', 'chair'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],])

    def __init__(self,
                 ann_file='SegmentationClassNpy',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        self.data_prefix = {
            'img_path': r'/root/FYP/data/JPEGImages',  # 图片路径
            'seg_map_path': r'/root/FYP/data/SegmentationClass'  # 分割标注路径
        }
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
