import os.path

import cv2
import numpy as np

import bb_stitcher.preperation as prep


def draw_makers(img, pts, color=(0, 0, 255), marker_types=cv2.MARKER_CROSS):
    img_m = np.copy(img)
    if len(img_m.shape) == 2:
        img_m = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pts = pts.astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types,
                       markerSize=40, thickness=5)
    return img_m


def test_Rectificator(left_img, config, outdir):
    rectificator = prep.Rectificator(config)
    corrected_image = rectificator.rectify_image(left_img['img'])
    assert corrected_image.shape == left_img['img'].shape

    # for visual see /out
    name_img_rect = ''.join([left_img['name'], '_rectified.jpg'])
    out = os.path.join(outdir, name_img_rect)
    cv2.imwrite(out, corrected_image)

    corrected_detections = rectificator.rectify_points(
        left_img['detections'], left_img['size'])
    assert len(corrected_detections) == len(left_img['detections'])

    # for visual see /out
    corrected_image_w_detections = rectificator.rectify_image(left_img['img_w_detections'])
    corrected_image_w_detections = draw_makers(corrected_image_w_detections, corrected_detections)
    name_img_rect_detec = ''.join([left_img['name'], '_detections_rectified.jpg'])
    out = os.path.join(outdir, name_img_rect_detec)
    cv2.imwrite(out, corrected_image_w_detections)


def test_rotate_image():
    img = np.zeros((30, 40), np.uint8)
    border = 2
    img[-border:, :] = 100  # bottom
    img[:, -border:] = 150  # right
    img[:border, :] = 200   # top
    img[:, :border] = 250   # left

    rot_img, mat = prep.rotate_image(img, 90)

    assert img.shape[0] == rot_img.shape[1] and img.shape[1] == rot_img.shape[0]

    assert rot_img[-1][15] == 250 or rot_img[-2][15] == 250  # bottom
    assert rot_img[20][-1] == 100 or rot_img[20][-2] == 100  # right
    assert rot_img[0][15] == 150 or rot_img[1][15] == 150  # top
    assert rot_img[20][0] == 200 or rot_img[20][1] == 200  # left

    rot_img, mat = prep.rotate_image(img, -90)
    assert img.shape[0] == rot_img.shape[1] and img.shape[1] == rot_img.shape[0]
    assert rot_img[-1][15] == 150 or rot_img[-2][15] == 150  # bottom
    assert rot_img[20][-1] == 200 or rot_img[20][-2] == 200  # right
    assert rot_img[0][15] == 250 or rot_img[1][15] == 250  # top
    assert rot_img[20][0] == 100 or rot_img[20][1] == 100  # left
