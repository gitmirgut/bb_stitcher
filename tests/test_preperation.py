import pytest
import bb_stitcher.preperation as prep


def test_Rectificator(img_hive_left, config):
    rectificator = prep.Rectificator(config)
    assert rectificator.rectify_images() is None
    rectificator.rectify_images(img_hive_left)