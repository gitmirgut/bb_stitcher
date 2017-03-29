import configparser
import os
import os.path

import cv2
import numpy as np
import pytest

import bb_stitcher.core as core


# add marker for incremental testing
# http://doc.pytest.org/en/latest/example/simple.html#incremental-testing-test-steps
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" %previousfailed.name)


test_dir = os.path.dirname(__file__)


def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)


def fname(path):
    return os.path.basename(os.path.splitext(path)[0])


@pytest.fixture
def left_img():
    img_path = 'data/Cam_0_2016-09-01T12:56:50.801920Z.jpg'
    img_with_detections = 'data/Cam_0_2016-09-01T12:56:50.801920Z_detections.jpg'
    detections_path = 'data/Cam_0_2016-09-01T12:56:50.801920Z_detections.npy'
    d = dict()
    d['path'] = get_test_fname(img_path)
    d['name'] = os.path.basename(os.path.splitext(d['path'])[0])
    d['img'] = cv2.imread(d['path'], -1)
    d['bw'] = cv2.imread(d['path'], 0)
    d['color'] = cv2.imread(d['path'], 1)
    d['height'], d['width'] = d['img'].shape[:2]
    d['size'] = d['width'], d['height']
    d['detections'] = np.load(get_test_fname(detections_path))
    d['img_w_detections'] = cv2.imread(get_test_fname(img_with_detections), -1)
    return d


@pytest.fixture
def right_img():
    img_path = 'data/Cam_1_2016-09-01T12:56:50.801926Z.jpg'
    img_with_detections = 'data/Cam_1_2016-09-01T12:56:50.801926Z_detections.jpg'
    detections_path = 'data/Cam_1_2016-09-01T12:56:50.801926Z_detections.npy'
    d = dict()
    d['path'] = get_test_fname(img_path)
    d['name'] = os.path.basename(os.path.splitext(d['path'])[0])
    d['img'] = cv2.imread(d['path'], -1)
    d['height'], d['width'] = d['img'].shape[:2]
    d['size'] = d['width'], d['height']
    d['detections'] = np.load(get_test_fname(detections_path))
    d['img_w_detections'] = cv2.imread(get_test_fname(img_with_detections), -1)
    return d


@pytest.fixture
def not_to_bee():
    img_path = get_test_fname('data/not_to_bee.jpg')
    img = cv2.imread(img_path, -1)
    return img


@pytest.fixture
def config():
    default_config = configparser.ConfigParser()
    default_config.read(core.get_default_config())
    return default_config


@pytest.fixture
def outdir():
    out_path = os.path.join(test_dir, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def out(filename):
    return os.path.join(outdir, filename)
