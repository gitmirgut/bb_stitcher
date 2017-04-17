import argparse

import pytest

import bb_stitcher.scripts.bb_stitcher as script_bb_stitcher
import bb_stitcher.io_utils as io_utils


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
