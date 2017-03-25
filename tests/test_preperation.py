import os.path

import cv2
import numpy as np
import numpy.testing as npt

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


def test_Rectificator(left_img, right_img, config, outdir):

    # left image
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
    rect_img = rectificator.rectify_image(left_img['img_w_detections'])
    marked_img = draw_makers(rect_img, corrected_detections)
    name_out = ''.join([left_img['name'], '_detections_rectified.jpg'])
    out = os.path.join(outdir, name_out)
    cv2.imwrite(out, marked_img)

    # right image
    corrected_image = rectificator.rectify_image(right_img['img'])
    assert corrected_image.shape == right_img['img'].shape

    # for visual see /out
    name_img_rect = ''.join([right_img['name'], '_rectified.jpg'])
    out = os.path.join(outdir, name_img_rect)
    cv2.imwrite(out, corrected_image)

    corrected_detections = rectificator.rectify_points(
        right_img['detections'], right_img['size'])
    assert len(corrected_detections) == len(right_img['detections'])

    # for visual see /out
    rect_img = rectificator.rectify_image(right_img['img_w_detections'])
    marked_img = draw_makers(rect_img, corrected_detections)
    name_out = ''.join([right_img['name'], '_detections_rectified.jpg'])
    out = os.path.join(outdir, name_out)
    cv2.imwrite(out, marked_img)


def test_rotate_image():
    img = np.zeros((30, 40), np.uint8)
    border = 1
    img[-border:, :] = 100  # bottom
    img[:, -border:] = 150  # right
    img[:border, :] = 200   # top
    img[:, :border] = 250   # left

    rot_img, mat = prep.rotate_image(img, 90)

    assert img.shape[0] == rot_img.shape[1] and img.shape[1] == rot_img.shape[0]

    assert rot_img[-1][15] == 250
    assert rot_img[20][-1] == 100
    assert rot_img[0][15] == 150
    assert rot_img[20][0] == 200

    rot_img, mat = prep.rotate_image(img, -90)
    assert img.shape[0] == rot_img.shape[1] and img.shape[1] == rot_img.shape[0]
    assert rot_img[-1][15] == 150
    assert rot_img[20][-1] == 200
    assert rot_img[0][15] == 250
    assert rot_img[20][0] == 100


def test_rotate_image_specifc(left_img, outdir):
    img, mat = prep.rotate_image(left_img['img'], 90)
    name_img_rot = ''.join([left_img['name'], '_rotated.jpg'])
    out = os.path.join(outdir, name_img_rot)
    cv2.imwrite(out, img)


def test_rotate_points():
    size = (4000, 3000)
    pts = np.array([
        [0, 0],
        [0, 2999],
        [3999, 2999],
        [3999, 0]
    ])

    rot_pts_pos90 = prep.rotate_points(pts, 90, size)
    target_points_pos90 = np.array([
        [0, 3999],
        [2999, 3999],
        [2999, 0],
        [0, 0]
    ])
    npt.assert_equal(rot_pts_pos90, target_points_pos90)

    rot_pts_neg90 = prep.rotate_points(pts, -90, size)
    target_points_neg90 = np.array([
        [2999, 0],
        [0, 0],
        [0, 3999],
        [2999, 3999]
    ])
    npt.assert_equal(rot_pts_neg90, target_points_neg90)


def test_rectify_and_rotate_image(left_img, config, outdir):
    rectificator = prep.Rectificator(config)
    rect_img = rectificator.rectify_image(left_img['img_w_detections'])
    rect_detections = rectificator.rectify_points(left_img['detections'], left_img['size'])

    angle = 90
    rot_img, rot_mat = prep.rotate_image(rect_img, angle)
    rot_detections = prep.rotate_points(rect_detections, angle, left_img['size'])

    marked_img = draw_makers(rot_img, rot_detections)
    name_out = ''.join([left_img['name'], '_detections_rectified_rot.jpg'])
    out = os.path.join(outdir, name_out)
    cv2.imwrite(out, marked_img)
