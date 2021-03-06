import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.helpers as helpers


def test_get_boundaries():
    size_left = (4, 3)
    size_right = (5, 2)
    homo_left = np.float32([[1, 0, -1],
                            [0, 1, 0],
                            [0, 0, 1]])
    homo_right = np.float32([[1, 0, 3],
                             [0, 1, -2],
                             [0, 0, 1]])
    bounds = helpers.get_boundaries(size_left, size_right, homo_left, homo_right)
    target_bounds = (-1, -2, 8, 3)
    assert bounds == target_bounds


def test_get_transform_to_origin_mat():
    homo = helpers.get_transform_to_origin_mat(-1, -3)
    target_homo = np.float32([
        [1, 0, 1],
        [0, 1, 3],
        [0, 0, 1]
    ])
    npt.assert_equal(homo, target_homo)
    homo = helpers.get_transform_to_origin_mat(1.3, -3)
    target_homo = np.float32([
        [1, 0, -1.3],
        [0, 1, 3],
        [0, 0, 1]
    ])
    npt.assert_equal(homo, target_homo)
    homo = helpers.get_transform_to_origin_mat(1, 3)
    target_homo = np.float32([
        [1, 0, -1],
        [0, 1, -3],
        [0, 0, 1]
    ])
    npt.assert_equal(homo, target_homo)
    homo = helpers.get_transform_to_origin_mat(-1, 3)
    target_homo = np.float32([
        [1, 0, 1],
        [0, 1, -3],
        [0, 0, 1]
    ])
    npt.assert_equal(homo, target_homo)


def test_add_alpha_channel(left_img):
    color = left_img['color']
    w, h = left_img['size']

    # test color image without alpha channel
    target = helpers.add_alpha_channel(color)
    assert target.shape == (h, w, 4)

    # test black and white image
    img_bw = left_img['bw']
    target = helpers.add_alpha_channel(img_bw)
    assert target.shape == (h, w, 4)

    # test already alpha
    img_alpha = np.zeros((3000, 4000, 4), dtype=np.uint8)
    helpers.add_alpha_channel(img_alpha)
    assert img_alpha.shape == (h, w, 4)

    # provoke exception
    img_not = np.zeros((3000, 4000, 5), dtype=np.uint8)
    with pytest.raises(Exception):
        helpers.add_alpha_channel(img_not)

    # provoke exception
    img_not = np.zeros((3000, 4000, 5, 4), dtype=np.uint8)
    with pytest.raises(Exception):
        helpers.add_alpha_channel(img_not)


def test_form_rectangle():
    height = 4
    width = 3
    target = np.array([
        [0, 0],
        [3, 0],
        [3, 4],
        [0, 4]
    ])
    result = helpers.form_rectangle(width, height)
    npt.assert_equal(result, target)


def test_sort_pts():
    # TODO(gitmirgut) Add more test.
    points = np.array([
        [2, 2],
        [2, 1],
        [1, 1],
        [1, 2]])
    target_points = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2]], dtype=np.float32)
    sorted_points = helpers.sort_pts(points)
    npt.assert_equal(sorted_points, target_points)


def test_raw_estimate_rect():
    points = np.array([
        [2, 2],
        [6, 1],
        [8, 8],
        [1, 7]
    ], dtype=np.float32)
    target_points = np.array([
        [0, 0],
        [7.0710, 0],
        [7.0710, 7.2801],
        [0, 7.2801]
    ])
    result = helpers.raw_estimate_rect(points)
    npt.assert_almost_equal(result, target_points, decimal=4)


def test_harmonize_rects():
    rect_a = np.array([
        [0, 0],
        [5, 0],
        [5, 4],
        [0, 4]
    ], dtype=np.float32)

    rect_b = np.array([
        [0, 0],
        [7, 0],
        [7, 6],
        [0, 6]
    ], dtype=np.float32)

    target_rect_a = np.array([
        [0, 0],
        [7.5, 0],
        [7.5, 6],
        [0, 6]
    ], dtype=np.float32)
    new_rect_a, new_rect_b = helpers.harmonize_rects(rect_a, rect_b)
    npt.assert_equal(new_rect_a, target_rect_a)
    npt.assert_equal(new_rect_b, rect_b)

    new_rect_b, new_rect_a = helpers.harmonize_rects(rect_b, rect_a)
    npt.assert_equal(new_rect_a, target_rect_a)
    npt.assert_equal(new_rect_b, rect_b)


def test_angles_to_points():
    angles = np.ones((5,))
    for i, val in enumerate(range(- 2, 3)):
        angles[i] = val * np.pi / 2

    points = np.zeros((5, 2), dtype=np.uint16)
    target = np.array([
        [-10, 0],
        [0, -10],
        [10, 0],
        [0, 10],
        [-10, 0]
    ])
    result = helpers.angles_to_points(points, angles, 10)
    npt.assert_almost_equal(result, target)

    points = np.ones((5, 2), dtype=np.uint16) * 10
    target = np.array([
        [0, 10],
        [10, 0],
        [20, 10],
        [10, 20],
        [0, 10]
    ])
    result = helpers.angles_to_points(points, angles, 10)
    npt.assert_almost_equal(result, target)


def test_points_to_angles(error_params):
    angle_centers = np.zeros((5, 2), dtype=np.uint16)
    points_repr = np.array([
        [-10, - 0.00000001],  # if it would be zero it will be pi
        [0, -10],
        [10, 0],
        [0, 10],
        [-10, 0]
    ])
    target = np.ones((5,))
    for i, val in enumerate(range(- 2, 3)):
        target[i] = val * np.pi / 2
    result = helpers.points_to_angles(angle_centers, points_repr)
    npt.assert_almost_equal(result, target, decimal=7)

    angle_centers = np.ones((1, 2))
    points_repr = np.ones((1, 2))
    with pytest.raises(Exception):
        helpers.points_to_angles(angle_centers, points_repr)

    # the following values are from a real bb_binary
    # hard to reconstruct with real values.
    angle_centers, points_repr = error_params

    target = np.array([0.])
    result = helpers.points_to_angles(angle_centers, points_repr)
    npt.assert_equal(target, result)


def test_get_ratio_px_to_mm():
    start_point = np.array([0, 0])
    end_point = np.array([30, 40])
    distance_mm = 25
    px_to_mm = helpers.get_ratio_px_to_mm(start_point, end_point, distance_mm)
    assert px_to_mm == 0.5


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
