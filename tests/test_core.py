import os

import cv2
import numpy as np
import pytest
from numpy import testing as npt

import bb_stitcher.core as core
import bb_stitcher.picking.picker
import bb_stitcher.helpers as helpers


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
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


@pytest.fixture
def fb_stitcher(config):
    fbs = core.FeatureBasedStitcher(config)
    return fbs


def test_get_default_config():
    config = helpers.get_default_config()

    config_set = {
        'Rectificator',
        'FeatureBasedStitcher',
        'SURF',
        'FeatureMatcher'}
    assert set(config.sections()) == config_set


def test_get_default_debug_config():
    deb_config = helpers.get_default_debug_config()
    assert 'loggers' in deb_config


"""
Test for feature based Stitcher.
"""


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
    mask_left, mask_right = core.FeatureBasedStitcher._calc_feature_mask(
        (3, 8), (3, 8), 2, 3, 1)
    npt.assert_equal(mask_left, target_mask_left)
    npt.assert_equal(mask_right, target_mask_right)


def test_fb_estimate_transform(fb_stitcher, left_img, right_img, not_to_bee):
    # provoke no finding of transformation
    assert fb_stitcher.estimate_transform(left_img['img'], not_to_bee) is None

    # find transformation
    assert fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None


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


"""
Test for rectangle Stitcher.
"""


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
    rt_stitcher = core.RectangleStitcher(config)
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
    rt_stitcher = core.RectangleStitcher(config)
    assert rt_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90) is not None
    pano = rt_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped = rt_stitcher.map_left_points(left_img['detections'])
    detections_right_mapped = rt_stitcher.map_right_points(right_img['detections'])
    pano = draw_marks(pano, detections_left_mapped)
    pano = draw_marks(pano, detections_right_mapped)

    out = os.path.join(outdir, 'panorama_rt_w_detections.jpg')
    cv2.imwrite(out, pano)
