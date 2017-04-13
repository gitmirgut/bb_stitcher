import os

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


@pytest.mark.incremental
class TestSurveyorDetermination:

    def test_determine_mapping_parameters(self, surveyor, left_img, right_img, monkeypatch):
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
        assert surveyor.pano_size[0] > 5500
        assert surveyor.pano_size[1] >= 4000
        assert 0 <= surveyor.origin[0] <= surveyor.pano_size[0]
        assert 0 <= surveyor.origin[1] <= surveyor.pano_size[1]
        assert surveyor._world_homo_left is not None
        assert surveyor._world_homo_right is not None

    def test_get_parameters(self, surveyor):
        result = surveyor.get_parameters()
        assert result.homo_left.shape == (3, 3)
        assert result.homo_right.shape == (3, 3)
        assert result.size_left == (4000, 3000)
        assert result.size_right == (4000, 3000)
        assert 5000 <= result.pano_size[0] <= 7000
        assert 3500 <= result.pano_size[1] <= 5000
        assert result.cam_id_left == 0
        assert result.cam_id_right == 1
        assert 0 <= result.origin[0] <= result.pano_size[0]
        assert 0 <= result.origin[1] <= result.pano_size[1]
        assert 0 < result.ratio_px_mm <= 1


@pytest.mark.incremental
class TestSurveyorMapping:
    def test_load_parameters(self, surveyor):
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
        pano_size = (5587, 4108)
        surveyor.load_parameters(homo_left, homo_right, size_left, size_right,
                                 cam_id_left, cam_id_right, origin, ratio_px_mm, pano_size)
        npt.assert_equal(surveyor.homo_left, homo_left)
        npt.assert_equal(surveyor.homo_right, homo_right)

        # after loading of parameters every instance variable should have a value that is not None
        for val in surveyor.__dict__.values():
            assert val is not None

    def test_map_points_angles(self, surveyor):
        pass
