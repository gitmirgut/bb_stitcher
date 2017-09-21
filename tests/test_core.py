import collections
import os

import cv2
import numpy as np
import numpy.testing as npt
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


@pytest.fixture(scope="class")
def surveyor(config):
    return core.Surveyor(config)


@pytest.fixture
def surveyor_params():
    StitchingParams = collections.namedtuple('SurveyorParams', ['homo_left', 'homo_right',
                                                                'size_left', 'size_right',
                                                                'cam_id_left', 'cam_id_right',
                                                                'origin', 'ratio_px_mm'])
    homo_left = np.array([
        [-1.98455538e-03, 1.02071571e+00, 5.95366897e+00],
        [-1.02273832e+00, 3.99220466e-03, 4.18182127e+03],
        [-4.53581281e-06, 1.47879389e-06, 1.01813872e+00]])
    homo_right = np.array([
        [4.15148897e-02, -1.03153036e+00, 5.58610125e+03],
        [1.03242057e+00, 1.02246208e-02, -1.37264940e-05],
        [6.53477459e-06, -8.57424632e-07, 1.00257142e+00]])
    size_left = (4000, 3000)
    size_right = (4000, 3000)
    cam_id_left = 0
    cam_id_right = 1
    origin = np.array([94.43029022, 471.89901733])
    ratio_px_mm = 0.0644410123918
    result = StitchingParams(homo_left, homo_right,
                             size_left, size_right,
                             cam_id_left, cam_id_right,
                             origin, ratio_px_mm)
    return result


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
    assert 0 <= surveyor.ratio_px_mm <= 1
    assert surveyor._world_homo_left is not None
    assert surveyor._world_homo_right is not None


@pytest.mark.incremental
class TestSurveyorGetSet:
    def test_set_parameters(self, surveyor, surveyor_params):
        surveyor.set_parameters(*surveyor_params)
        npt.assert_equal(surveyor.homo_left, surveyor.homo_left)
        npt.assert_equal(surveyor.homo_right, surveyor.homo_right)

        # after loading of parameters every instance variable should have a value that is not None
        for val in surveyor.__dict__.values():
            assert val is not None

    def test_get_parameters(self, surveyor, surveyor_params):
        result = surveyor.get_parameters()
        npt.assert_equal(result, surveyor_params)


@pytest.mark.incremental
class TestSurveyorMapping:

    def test_map_points_angles_left(self, surveyor, surveyor_params, left_img):
        surveyor.set_parameters(*surveyor_params)
        with pytest.raises(ValueError):
            # bad cam id should raise error
            surveyor.map_points_angles(left_img['detections'], left_img['yaw_angles'], 666)

        points, angles = surveyor.map_points_angles(left_img['detections'], left_img['yaw_angles'],
                                                    left_img['cam_id'])

        # left points must have an x coordinate lower then circa 2/3 of the width of the comb
        assert np.max(points[:, 0]) <= 220
        assert -50 <= np.min(points[:, 0])

        assert np.max(points[:, 1]) <= 250
        assert -50 <= np.min(points[:, 0])

        assert np.max(angles) <= np.pi
        assert -np.pi <= np.min(angles)

    def test_map_points_angles_right(self, surveyor, right_img):
        points, angles = surveyor.map_points_angles(right_img['detections'],
                                                    right_img['yaw_angles'],
                                                    right_img['cam_id'])

        assert np.max(points[:, 0]) <= 360
        assert 150 <= np.min(points[:, 0])

        assert np.max(points[:, 1]) <= 250
        assert -50 <= np.min(points[:, 0])

        assert np.max(angles) <= np.pi
        assert -np.pi <= np.min(angles)

    def test_compose_panorama(self, surveyor, left_img, right_img, outdir):
        pano = surveyor.compose_panorama(left_img['path'], right_img['path'], grid=True)
        assert pano.shape[0] >= 4000
        assert pano.shape[1] >= 5000
        out = os.path.join(outdir, 'panorama_w_grid.jpg')
        cv2.imwrite(out, pano)

    def test_save(self, surveyor, outdir):
        out = os.path.join(outdir, 'Surveyor_data.npz')
        surveyor.save(out)


@pytest.fixture(scope="class", params=['.npz', '.csv', '.json'])
def ext(request):
    return request.param


@pytest.mark.incremental
class TestSurveyorFileHandlerNPZ:

    @classmethod
    @staticmethod
    @pytest.fixture(scope="function")
    def surveyor(config):
        return core.Surveyor(config)

    def test_save(self, surveyor, surveyor_params, ext, outdir):
        surveyor.set_parameters(*surveyor_params)
        out = os.path.join(outdir, 'Surveyor_data')
        out_w_ext = ''.join([out, ext])
        surveyor.save(out_w_ext)

    def test_load(self, surveyor, surveyor_params, ext, outdir):
        input = os.path.join(outdir, 'Surveyor_data')
        input_w_ext = ''.join([input, ext])
        surveyor.load(input_w_ext)
        npt.assert_equal(surveyor.get_parameters(), surveyor_params)


def test_bad_extension_FileHandler(surveyor):
    with pytest.raises(Exception):
        surveyor.load('not_to_bee.666')
