import os

import cv2
import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.core as core
import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker
import bb_stitcher.prep as prep
import bb_stitcher.stitcher as stitcher
import bb_stitcher.visualisation as vis


@pytest.fixture
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


@pytest.fixture
def fb_stitcher(config):
    fbs = core.FeatureBasedStitcher(config)
    return fbs


def draw_marks(img, pts, color=(0, 0, 255), marker_types=cv2.MARKER_CROSS):
    img_m = np.copy(img)
    if len(img_m.shape) == 2:
        img_m = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pts = pts.astype(int)
    for pt in pts:
        cv2.drawMarker(img_m, tuple(pt), color, markerType=marker_types,
                       markerSize=40, thickness=5)
    return img_m


@pytest.fixture
def super_stitcher(config):
    st = stitcher.Stitcher(config)
    return st


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


@pytest.fixture
def homo_left():
    homo = np.float64([
        [1, 0, 0],
        [0, 1, 199.91238403],
        [0, 0, 1]
    ])
    return homo


@pytest.fixture
def homo_right():
    homo = np.float64([
        [1.01332402e+00, 4.11682445e-02, 2.46059578e+03],
        [-4.11682445e-02, 1.01332402e+00, 1.23504729e+02],
        [0, 0, 1]
    ])
    return homo


@pytest.fixture
def pano_size():
    return (5666, 4200)


def test_super_estimate_transform(super_stitcher):
    with pytest.raises(NotImplementedError):
        super_stitcher.estimate_transform(None, None)


def test_prepare_image(left_img, super_stitcher):
    super_stitcher._prepare_image(left_img['img'])


def test_map_points(config):
    homo_left = np.array(
        [[1, 0, 2],
         [0, 1, 1],
         [0, 0, 1]])
    homo_right = np.array(
        [[1, 0, 1],
         [0, 1, 2],
         [0, 0, 1]])
    points = np.array(
        [[1, 1],
         [2, 2],
         [3, 3]], dtype=np.float32)
    target_left = np.array(
        [[3, 2],
         [4, 3],
         [5, 4]])
    target_right = np.array(
        [[2, 3],
         [3, 4],
         [4, 5]])
    size = (4000, 3000)
    st = stitcher.Stitcher(config, rectify=False)
    st.load_parameters(homo_left, homo_right, size, size)
    pano_points_left = st.map_left_points(points)
    pano_points_right = st.map_right_points(points)

    npt.assert_equal(pano_points_left, target_left)
    npt.assert_equal(pano_points_right, target_right)


def test_map_points_angles(left_img, right_img, outdir, config, monkeypatch):
    def mockreturn(myself, image_list, all):
        left_points = np.array([
            [88.91666412, 3632.6015625],
            [2760.26855469, 3636.70849609],
            [2726.26708984, 363.4861145],
            [93.88884735, 371.98330688]], dtype=np.float32)
        right_points = np.array([
            [181.49372864, 3687.71582031],
            [2903.44042969, 3723.99926758],
            [2921.27368164, 458.77352905],
            [255.66642761, 431.24780273]], dtype=np.float32)
        return left_points, right_points
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mockreturn)
    rt_stitcher = core.RectangleStitcher(config)
    assert rt_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None
    pano = rt_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped, yaw_angles_left_mapped = rt_stitcher.map_left_points_angles(
        left_img['detections'], left_img['yaw_angles'])
    detections_right_mapped, yaw_angles_right_mapped = rt_stitcher.map_right_points_angles(
        right_img['detections'], right_img['yaw_angles'])
    vis.draw_complex_marks(pano, detections_left_mapped, yaw_angles_left_mapped)
    vis.draw_complex_marks(pano, detections_right_mapped, yaw_angles_right_mapped)
    out = os.path.join(outdir, 'panorama_rt_w_detections_angles.jpg')
    cv2.imwrite(out, pano)


@pytest.mark.slow
def test_compose_panorama(fb_stitcher, left_img, right_img, outdir):
    fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    pano = fb_stitcher.compose_panorama(left_img['img'], right_img['img'])
    out = os.path.join(outdir, 'panorama.jpg')
    cv2.imwrite(out, pano)

def test_calc_image_to_world_mat(super_stitcher, panorma, monkeypatch):
    def mockreturn(myself, image_list, all):
        points = [np.array([
            [94.43029022,   471.89901733],
            [5494.71777344,   471.83984375]
        ], dtype=np.float32)]
        return points
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mockreturn)
    monkeypatch.setitem(__builtins__, 'input', lambda x: "348")
    super_stitcher._calc_image_to_world_mat(panorma)
