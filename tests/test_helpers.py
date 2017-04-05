import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.helpers as helpers


def test_align_to_display_area():
    # TODO(gitmirgut): Add test with rotation.
    size_left = (4, 3)
    size_right = (5, 2)
    homo_left = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    homo_right = np.float32([[1, 0, 3], [0, 1, 2], [0, 0, 1]])
    homo_trans, display_size = helpers.align_to_display_area(
        size_left, size_right, homo_left, homo_right)

    target_homo = np.float32(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])
    target_size = (8, 4)

    npt.assert_equal(homo_trans, target_homo)
    assert display_size == target_size


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


def test_points_to_angles():
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
