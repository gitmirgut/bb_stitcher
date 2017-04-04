import os

import cv2
import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker
import bb_stitcher.prep as prep
import bb_stitcher.stitcher as stitcher


@pytest.fixture
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


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
def fb_stitcher(config):
    fbs = stitcher.FeatureBasedStitcher(config)
    return fbs


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


def test_calc_feature_mask():
    target_mask_left = np.array(
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 255, 255],
         [0, 255, 255],
         [0, 255, 255],
         [0, 255, 255],
         [0, 0, 0],
         ], np.uint8)
    target_mask_right = np.array(
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [255, 255, 0],
         [255, 255, 0],
         [255, 255, 0],
         [255, 255, 0],
         [0, 0, 0],
         ], np.uint8)
    mask_left, mask_right = stitcher.FeatureBasedStitcher._calc_feature_mask(
        (3, 8), (3, 8), 2, 3, 1)
    npt.assert_equal(mask_left, target_mask_left)
    npt.assert_equal(mask_right, target_mask_right)


@pytest.fixture()
def test_fb_estimate_transform(fb_stitcher, left_img, right_img, not_to_bee):
    # provoke no finding of transformation
    assert fb_stitcher.estimate_transform(left_img['img'], not_to_bee) is None

    # find transformation
    assert fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None


def test_prepare_image(left_img, super_stitcher):
    super_stitcher._prepare_image(left_img['img'])


@pytest.mark.slow
def test_compose_panorama(fb_stitcher, left_img, right_img, outdir):
    fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    pano = fb_stitcher.compose_panorama(left_img['img'], right_img['img'])
    out = os.path.join(outdir, 'panorama.jpg')
    cv2.imwrite(out, pano)


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
    st = stitcher.Stitcher(config, homo_left, homo_right, size, size, rectify=False)
    pano_points_left = st.map_left_points(points)
    pano_points_right = st.map_right_points(points)

    npt.assert_equal(pano_points_left, target_left)
    npt.assert_equal(pano_points_right, target_right)


@pytest.mark.slow
def test_overall_fb_stitching(fb_stitcher, left_img, right_img, outdir):
    assert fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None
    pano = fb_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped = fb_stitcher.map_left_points(left_img['detections'])
    detections_right_mapped = fb_stitcher.map_right_points(right_img['detections'])
    pano = draw_marks(pano, detections_left_mapped)
    pano = draw_marks(pano, detections_right_mapped)

    out = os.path.join(outdir, 'panorama_fb_w_detections.jpg')
    cv2.imwrite(out, pano)


def test_rect_stitcher_estimate_transform(left_img, right_img, outdir, config,
                                          monkeypatch):
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
    # print(left_points)
    rt_stitcher = stitcher.RectangleStitcher(config)
    homo_left, homo_right, pano_size = rt_stitcher.estimate_transform(
        left_img['img'], right_img['img'], 90, -90)
    assert homo_left is not None
    assert homo_right is not None
    assert pano_size is not None


@pytest.mark.slow
def test_overall_rt_stitching(left_img, right_img, outdir, config, monkeypatch):
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
    rt_stitcher = stitcher.RectangleStitcher(config)
    assert rt_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None
    pano = rt_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped = rt_stitcher.map_left_points(left_img['detections'])
    detections_right_mapped = rt_stitcher.map_right_points(right_img['detections'])
    pano = draw_marks(pano, detections_left_mapped)
    pano = draw_marks(pano, detections_right_mapped)

    out = os.path.join(outdir, 'panorama_rt_w_detections.jpg')
    cv2.imwrite(out, pano)
