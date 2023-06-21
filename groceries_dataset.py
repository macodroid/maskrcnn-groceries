import os
import random

import numpy as np
import torch
from pycocotools.coco import COCO
import cv2 as cv
from torch.utils.data import Dataset
import config_maskrcnn as cfg


class GroceriesDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform='train'):
        self.root_dir = root_dir
        self.ann_file = ann_file
        self.transform = transform
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img = cv.imread(os.path.join(self.root_dir, path))

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            x_min = coco_annotation[i]["bbox"][0]
            y_min = coco_annotation[i]["bbox"][1]
            x_max = x_min + coco_annotation[i]["bbox"][2]
            y_max = y_min + coco_annotation[i]["bbox"][3]
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        masks = []
        areas = []
        for i in range(num_objs):
            masks.append(coco.annToMask(coco_annotation[i]))
            areas.append(coco_annotation[i]["area"])
        masks = np.stack(masks, axis=0)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transform == 'train':
            img = cfg.train_transforms(img)
        else:
            img = cfg.test_transforms(img)

        my_annotation = {"boxes": boxes, "labels": labels, "masks": torch.tensor(masks),
                         "image_id": torch.tensor([img_id]), "area": torch.as_tensor(areas), "iscrowd": iscrowd}
        return img, my_annotation


def collate_fn(batch):
    return tuple(zip(*batch))
