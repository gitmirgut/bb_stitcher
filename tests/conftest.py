import pytest
import os
import cv2
import bb_stitcher.core as core
import configparser



def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)

@pytest.fixture()
def config():
    default_config = configparser.ConfigParser()
    default_config.read(core.get_default_config())
    return default_config

@pytest.fixture
def img_hive_left():
    path = get_test_fname('data/Cam_0_2016-09-01T12:56:50.801920Z.jpg')
    return cv2.imread(path, -1)