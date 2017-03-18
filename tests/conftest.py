import pytest
import os
import cv2
import bb_stitcher.core as core
import configparser
import os.path
import numpy as np

test_dir = os.path.dirname(__file__)

def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)

def fname(path):
    return os.path.basename(os.path.splitext(path)[0])

@pytest.fixture()
def config():
    default_config = configparser.ConfigParser()
    default_config.read(core.get_default_config())
    return default_config

@pytest.fixture
def img_left_path():
    path = get_test_fname('data/Cam_0_2016-09-01T12:56:50.801920Z.jpg')
    return path

@pytest.fixture
def detections_left_img():
    path = get_test_fname('data/Cam_0_2016-09-01T12:56:50.801920Z_detections.npy')
    return np.load(path)

@pytest.fixture
def outdir():
    out_path = os.path.join(test_dir, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path

def out(filename):
    return os.path.join(outdir, filename)