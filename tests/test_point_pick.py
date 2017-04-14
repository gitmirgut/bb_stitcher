import numpy as np
import pytest

import bb_stitcher.helpers as helpers
import bb_stitcher.prep as prep
import bb_stitcher.picking.picker as picker


def create_prepared_image_dict(img, angle, config):
    img_alpha = helpers.add_alpha_channel(img['img'])
    rectificator = prep.Rectificator(config)
    rect_img = rectificator.rectify_image(img_alpha)
    rect_detections = rectificator.rectify_points(img['detections'], img['size'])
    rect_img_w_detections = rectificator.rectify_image(img['img_w_detections'])

    rot_img, rot_mat = prep.rotate_image(rect_img, angle)
    rot_detections = prep.rotate_points(rect_detections, angle, img['size'])
    rot_img_w_detections, rot_mat = prep.rotate_image(rect_img_w_detections, angle)

    d = dict()
    d['img'] = rot_img
    d['detections'] = rot_detections
    d['img_w_detections'] = rot_img_w_detections
    return d


@pytest.fixture
def left_img_prep(left_img, config):
    return create_prepared_image_dict(left_img, 90, config)


@pytest.fixture
def right_img_prep(right_img, config):
    return create_prepared_image_dict(right_img, -90, config)


@pytest.mark.slow
def test_gui(left_img_prep, right_img_prep):
    # TODO(gitmirgut): better gui test...
    pt = picker.PointPicker()
    print(pt)
    # points = pt.pick([left_img_prep['img'], right_img_prep['img']])
    # print(points)


def test_pick_length(panorama, monkeypatch):
    def mockreturn(myself, image_list, all):
        points = np.array([
            [94.43035126, 471.89889526],
            [5494.71777344, 471.83984375]], dtype=np.float32)
        return points
    monkeypatch.setattr(picker.PointPicker, 'pick', mockreturn)
    pt = picker.PointPicker()
    points = pt.pick([panorama], False)
    assert len(points) == 2
    start_point, end_point = points
    print(start_point.shape)
    distance_px = np.linalg.norm(end_point - start_point)
    distance_mm = 344
    px_to_mm = distance_mm / distance_px
    print(distance_px * px_to_mm)
