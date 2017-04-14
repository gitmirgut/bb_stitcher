import os

import cv2
import numpy as np
import pytest
import numpy.testing as npt

import bb_stitcher.picking.picker
import bb_stitcher.stitcher as stitcher
import bb_stitcher.visualisation as vis


@pytest.fixture
def outdir(main_outdir):
    # out_path = os.path.join(main_outdir, str(__name__))
    out_path = os.path.join(main_outdir, 'test_stitcher2')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


@pytest.fixture
def super_stitcher(config):
    st = stitcher.Stitcher(config)
    return st


@pytest.fixture
def fb_stitcher(config):
    fbs = stitcher.FeatureBasedStitcher(config)
    return fbs


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

    rt_stitcher = stitcher.RectangleStitcher(config)
    rt_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    assert rt_stitcher.homo_left is not None
    assert rt_stitcher.homo_right is not None
    assert rt_stitcher.pano_size is not None
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


def test_calc_image_to_world_mat(super_stitcher, panorama, monkeypatch):
    def mockreturn(myself, image_list, all):
        points = [np.array([
            [94.43029022, 471.89901733],
            [5494.71777344, 471.83984375]
        ], dtype=np.float32)]
        return points
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mockreturn)
    monkeypatch.setitem(__builtins__, 'input', lambda x: "348")

    super_stitcher._calc_image_to_world_mat(panorama)


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


def test_fb_estimate_transform(fb_stitcher, left_img, right_img, not_to_bee):
    # provoke no finding of transformation
    fb_stitcher.estimate_transform(left_img['img'], not_to_bee)
    assert fb_stitcher.homo_left is None and fb_stitcher.homo_right is None

    # find transformation
    fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    assert fb_stitcher.homo_left is not None and fb_stitcher.homo_right is not None


@pytest.mark.slow
def test_overall_fb_stitching(fb_stitcher, left_img, right_img, outdir):
    fb_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    assert fb_stitcher.homo_left is not None and fb_stitcher.homo_right is not None
    pano = fb_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped = fb_stitcher.map_left_points(left_img['detections'])
    detections_right_mapped = fb_stitcher.map_right_points(right_img['detections'])
    vis.draw_marks(pano, detections_left_mapped)
    vis.draw_marks(pano, detections_right_mapped)

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

    rt_stitcher = stitcher.RectangleStitcher(config)
    rt_stitcher.estimate_transform(
        left_img['img'], right_img['img'], 90, -90)
    assert rt_stitcher.homo_left is not None
    assert rt_stitcher.homo_right is not None
    assert rt_stitcher.pano_size is not None


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
    rt_stitcher.estimate_transform(left_img['img'], right_img['img'], 90, -90)
    assert rt_stitcher.homo_left is not None
    assert rt_stitcher.homo_right is not None
    assert rt_stitcher.pano_size is not None
    pano = rt_stitcher.compose_panorama(
        left_img['img_w_detections'], right_img['img_w_detections'])
    detections_left_mapped = rt_stitcher.map_left_points(left_img['detections'])
    detections_right_mapped = rt_stitcher.map_right_points(right_img['detections'])
    vis.draw_marks(pano, detections_left_mapped)
    vis.draw_marks(pano, detections_right_mapped)

    out = os.path.join(outdir, 'panorama_rt_w_detections.jpg')
    cv2.imwrite(out, pano)
