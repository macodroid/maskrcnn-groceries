import numpy as np
import cv2 as cv
import json

def load_annotations(image_path: str):
    annotation_path = image_path.replace(".png", ".coco.json")
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    return annotations

def get_image_and_seg_points(image_path: str):
    image = cv.imread(image_path)
    annotations = load_annotations(image_path)
    return image, np.reshape(np.array(annotations['annotations'][0]["segmentation"], dtype=np.int32), (-1, 2))


if __name__ == "__main__":
