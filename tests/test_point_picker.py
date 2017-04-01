import pytest

import bb_stitcher.helpers as helpers
import bb_stitcher.preparation as prep
import bb_stitcher.point_picker as point_picker


@pytest.fixture
def left_img_prep(left_img, config):
    left_img_alpha = helpers.add_alpha_channel(left_img['img'])
    rectificator = prep.Rectificator(config)
    rect_img = rectificator.rectify_image(left_img_alpha)
    rect_detections = rectificator.rectify_points(left_img['detections'], left_img['size'])
    rect_img_w_detections = rectificator.rectify_image(left_img['img_w_detections'])

    angle = 90
    rot_img, rot_mat = prep.rotate_image(rect_img, angle)
    rot_detections = prep.rotate_points(rect_detections, angle, left_img['size'])
    rot_img_w_detections, rot_mat = prep.rotate_image(rect_img_w_detections, angle)

    d = dict()
    d['img'] = rot_img
    d['detections'] = rot_detections
    d['img_w_detections'] = rot_img_w_detections

    return d


@pytest.fixture
def right_img_prep(right_img, config):
    right_img_alpha = helpers.add_alpha_channel(right_img['img'])
    rectificator = prep.Rectificator(config)
    rect_img = rectificator.rectify_image(right_img_alpha)
    rect_detections = rectificator.rectify_points(right_img['detections'], right_img['size'])
    rect_img_w_detections = rectificator.rectify_image(right_img['img_w_detections'])

    angle = -90
    rot_img, rot_mat = prep.rotate_image(rect_img, angle)
    rot_detections = prep.rotate_points(rect_detections, angle, right_img['size'])
    rot_img_w_detections, rot_mat = prep.rotate_image(rect_img_w_detections, angle)

    d = dict()
    d['img'] = rot_img
    d['detections'] = rot_detections
    d['img_w_detections'] = rot_img_w_detections

    return d


@pytest.mark.slow
def test_point_picker(left_img_prep, right_img_prep):
    pt = point_picker.PointPicker(left_img_prep['img'], left_img_prep['img'])
    print(pt)
    points = pt.pick()
    print(points)
