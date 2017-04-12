import os

import numpy as np
import pytest

import bb_stitcher.core as core
import bb_stitcher.stitcher as stitcher
import bb_stitcher.picking.picker


@pytest.fixture
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


@pytest.fixture
def surveyor(config):
    return core.Surveyor(config)


def test_determine_mapping_parameters(surveyor, left_img, right_img, monkeypatch):
    def mock_pick(myself, image_list, all):
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

    def mock_get_origin(image):
        return np.array([94.43029022, 471.89901733])

    def mock_get_ratio(image):
        return 0.0644410123918
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mock_pick)
    monkeypatch.setattr(bb_stitcher.measure, 'get_origin', mock_get_origin)
    monkeypatch.setattr(bb_stitcher.measure, 'get_ratio', mock_get_ratio)
    surveyor.determine_mapping_parameters(left_img['path'], right_img['path'],
                                          90, -90,
                                          0, 1,
                                          stitcher.RectangleStitcher)
