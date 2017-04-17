import argparse
import os
import sys

import numpy as np
import pytest

import bb_stitcher.measure
import bb_stitcher.picking
import bb_stitcher.scripts.bb_stitcher as script_bb_stitcher
import bb_stitcher.io_utils as io_utils


@pytest.fixture
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def test_exist_path():
    with pytest.raises(argparse.ArgumentTypeError):
        script_bb_stitcher._exist_path('move_along_no_file_to_see_here')

    assert __file__ == script_bb_stitcher._exist_path(__file__)


def test_data_path():
    for ext in io_utils.valid_ext:
        output_file = ''.join(['file', ext])
        assert output_file == script_bb_stitcher._data_path(output_file)

    with pytest.raises(argparse.ArgumentTypeError):
        script_bb_stitcher._data_path('file_with_bad_extension.666')


def test_img_path():
    for ext in ['.jpeg', '.jpg', '.png']:
        img_path = ''.join(['img', ext])
        assert img_path == script_bb_stitcher._img_path(img_path)

    with pytest.raises(argparse.ArgumentTypeError):
        script_bb_stitcher._img_path('eps2.9_pyth0n-pt1.p7z')


def test_get_main_parser(left_img, right_img, surveyor_csv_path):
    # todo(gitmirgut) better test on bad values see
    # http://stackoverflow.com/questions/18651705/argparse-unit-tests-suppress-the-help-message#18652005
    main_parser = script_bb_stitcher._get_main_parser()
    cmd = 'estimate fb {left_img} {right_img} 90 -90 0 1  test.csv'.format(
        left_img=left_img['path'],
        right_img=right_img['path']
    )
    main_parser.parse_args(cmd.split())

    cmd = 'estimate rect {left_img} {right_img} 90 -90 0 1  test.csv'.format(
        left_img=left_img['path'],
        right_img=right_img['path']
    )
    main_parser.parse_args(cmd.split())

    cmd = 'compose {left_img} {right_img} {data_path} pano.jpg'.format(
        left_img=left_img['path'],
        right_img=right_img['path'],
        data_path=surveyor_csv_path
    )
    main_parser.parse_args(cmd.split())


def test_overall_compose(left_img, right_img, surveyor_csv_path, outdir, monkeypatch):
    out = os.path.join(outdir, 'panorama_compose.jpg')
    cmd = 'bb_stitcher compose {left} {right} {data} {out} -g'.format(left=left_img['path'],
                                                                      right=right_img['path'],
                                                                      data=surveyor_csv_path,
                                                                      out=out
                                                                      )
    monkeypatch.setattr(sys, 'argv', cmd.split())
    script_bb_stitcher.main()
    assert os.path.exists(out)


def test_overall_estimate_rect(left_img, right_img, outdir, monkeypatch):
    out = os.path.join(outdir, 'data_rect.csv')
    cmd = 'bb_stitcher estimate rect {left} {right} {left_angle} {right_angle} ' \
          '{left_camId} {right_camId} {out}'.format(left=left_img['path'],
                                                    right=right_img['path'],
                                                    left_angle=left_img['angle'],
                                                    right_angle=right_img['angle'],
                                                    left_camId=left_img['cam_id'],
                                                    right_camId=right_img['cam_id'],
                                                    out=out)

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

    monkeypatch.setattr(sys, 'argv', cmd.split())
    monkeypatch.setattr(bb_stitcher.picking.picker.PointPicker, 'pick', mock_pick)
    monkeypatch.setattr(bb_stitcher.measure, 'get_origin', mock_get_origin)
    monkeypatch.setattr(bb_stitcher.measure, 'get_ratio', mock_get_ratio)
    script_bb_stitcher.main()
    assert os.path.exists(out)


def test_overall_estimate_fb(left_img, right_img, outdir, monkeypatch):
    out = os.path.join(outdir, 'data_fb.npz')
    cmd = 'bb_stitcher estimate fb {left} {right} {left_angle} {right_angle} ' \
          '{left_camId} {right_camId} {out}'.format(left=left_img['path'],
                                                    right=right_img['path'],
                                                    left_angle=left_img['angle'],
                                                    right_angle=right_img['angle'],
                                                    left_camId=left_img['cam_id'],
                                                    right_camId=right_img['cam_id'],
                                                    out=out)

    def mock_get_origin(image):
        return np.array([94.43029022, 471.89901733])

    def mock_get_ratio(image):
        return 0.0644410123918

    monkeypatch.setattr(sys, 'argv', cmd.split())
    monkeypatch.setattr(bb_stitcher.measure, 'get_origin', mock_get_origin)
    monkeypatch.setattr(bb_stitcher.measure, 'get_ratio', mock_get_ratio)
    script_bb_stitcher.main()
    assert os.path.exists(out)
