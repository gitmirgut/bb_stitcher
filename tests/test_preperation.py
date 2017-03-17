import pytest
import bb_stitcher.preperation as prep
import cv2
import os.path

def test_Rectificator(img_hive_left, config, outdir):
    rectificator = prep.Rectificator(config)
    corrected_image = rectificator.rectify_image(img_hive_left)
    out = os.path.join(outdir, 'correct_img_hive_left.jpg')
    cv2.imwrite(out, corrected_image)