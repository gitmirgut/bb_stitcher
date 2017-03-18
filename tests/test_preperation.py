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

def test_Rectificator(img_left_path, config, outdir, detections_left_img):
    img = cv2.imread(img_left_path, -1)
    bname_img = bname(img_left_path, 'rectified', 'jpg')
    rectificator = prep.Rectificator(config)
    corrected_image = rectificator.rectify_image(img)
    out = os.path.join(outdir, bname_img)
    cv2.imwrite(out, corrected_image)

    corrected_points = rectificator.rectify_points(detections_left_img, 3000, 4000)
    for pos in corrected_points:
        pos = pos.astype(np.int32)
        draw_circle(corrected_image, pos)
    bname_img_detections = bname(img_left_path, 'rectified_detections', 'jpg')
    out_detections = os.path.join(outdir, bname_img_detections)
    cv2.imwrite(out_detections, corrected_image)
    assert corrected_points.shape == detections_left_img.shape