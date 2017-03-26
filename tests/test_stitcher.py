import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.preperation as prep
import bb_stitcher.stitcher as stitcher


@pytest.fixture()
def fb_stitcher(config):
    fbs = stitcher.FeatureBasedStitcher(config)
    return fbs


@pytest.fixture()
def left_img_prep(left_img, config):
    rect = prep.Rectificator(config)

    prepared_img = rect.rectify_image(left_img['img'])
    prepared_img, affine = prep.rotate_image(prepared_img, 90)
    return prepared_img


@pytest.fixture()
def right_img_prep(right_img, config):
    rect = prep.Rectificator(config)

    prepared_img = rect.rectify_image(right_img['img'])
    prepared_img, affine = prep.rotate_image(prepared_img, -90)
    return prepared_img


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


def test_estimate_transformation(fb_stitcher, left_img_prep, right_img_prep):
    fb_stitcher.estimate_transformation(left_img_prep, right_img_prep)
