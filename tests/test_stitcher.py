import numpy as np
import numpy.testing as npt
import pytest

import bb_stitcher.stitcher as stitcher


@pytest.fixture()
def fb_stitcher(config):
    fbs = stitcher.FeatureBasedStitcher(config)
    return fbs


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


def test_estimate_transformation(fb_stitcher, left_img, right_img):
    fb_stitcher.estimate_transformation(left_img['img'], right_img['img'])
