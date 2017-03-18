import pytest
import bb_stitcher.preperation as prep
import cv2
import os.path
import numpy as np

def draw_circle(img, position, radius=32, line_width=6, color=(0, 0, 255)):
    """Draw circle around position."""
    cv2.circle(img, tuple(position), radius, color, line_width)

def bname(path, discription, ext):
    basename_no_ext = os.path.basename(os.path.splitext(path)[0])
    return ''.join([basename_no_ext, '_', discription,'.', ext])

def fname(path):
    return os.path.basename(os.path.splitext(path)[0])

def test_Rectificator(left_img, config, outdir):
    rectificator = prep.Rectificator(config)
    corrected_image = rectificator.rectify_image(left_img['img'])
    assert corrected_image.shape == left_img['img'].shape
    name_img_rect = ''.join([left_img['name'], '_rectified.jpg'])
    out = os.path.join(outdir, name_img_rect)
    cv2.imwrite(out, corrected_image)

    corrected_detections = rectificator.rectify_points(left_img['detections'], left_img['height'], left_img['width'])
    assert len(corrected_detections) == len(left_img['detections'])
    for pos in corrected_detections:
        pos = pos.astype(np.int32)
        draw_circle(corrected_image, pos)
    name_img_rect_detec = ''.join([left_img['name'], '_rectified_detections.jpg'])
    out = os.path.join(outdir, name_img_rect_detec)
    cv2.imwrite(out, corrected_image)

