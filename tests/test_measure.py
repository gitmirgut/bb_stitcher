import numpy as np
import numpy.testing as npt

import bb_stitcher.picking.picker
import bb_stitcher.measure as measure


def test_get_ratio(panorama, monkeypatch):
    def mockreturn(myself, image_list, all):
        points = [np.array([
            [94.43029022, 471.89901733],
            [5494.71777344, 471.83984375]
        ], dtype=np.float32)]
        return points
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mockreturn)
    monkeypatch.setitem(__builtins__, 'input', lambda x: "348")
    ratio = measure.get_ratio(panorama)
    target = 348 / 5400
    npt.assert_almost_equal(ratio, target, decimal=4)


def test_get_origin(panorama, monkeypatch):
    def mockreturn(myself, image_list, all):
        points = [np.array([
            [94.43029022, 471.89901733]
        ], dtype=np.float32)]
        return points
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mockreturn)
    point = measure.get_origin(panorama)
    assert point is not None
