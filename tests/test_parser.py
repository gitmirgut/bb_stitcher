import argparse

import pytest

import bb_stitcher.io_utils as io_utils
import bb_stitcher.scripts._parser as parser


def test_filepath():
    with pytest.raises(argparse.ArgumentTypeError):
        parser.exist_path('move_along_no_file_to_see_here')

    assert __file__ == parser.exist_path(__file__)


def test_output_path():
    for ext in io_utils.valid_ext:
        output_file = ''.join(['file', ext])
        assert output_file == parser.data_path(output_file)

    with pytest.raises(argparse.ArgumentTypeError):
        parser.data_path('file_with_bad_extension.666')


def test_get_parser(left_img, right_img):
    # todo(gitmirgut) better test on bad values see
    # http://stackoverflow.com/questions/18651705/argparse-unit-tests-suppress-the-help-message#18652005
    main_parser = parser.get_parser()
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

    # cmd = 'compose {left_img} {right_img} test.csv pano.jpg'.format(
    #     left_img=left_img['path'],
    #     right_img=right_img['path']
    # )
    # main_parser.parse_args(cmd.split())
