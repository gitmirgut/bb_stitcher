import argparse

import pytest

import bb_stitcher.scripts.cli as cli


@pytest.fixture()
def parser():
    ret = argparse.ArgumentParser(
        prog='cli tester',
        description='This parser is for testing of the cli module',
        formatter_class=argparse.RawTextHelpFormatter
    )
    return ret


def test_add_shared_positional_arguments(parser):
    cli.add_shared_positional_arguments(parser)
    cli.add_shared_optional_arguments(parser)
    args = parser.parse_args(['left_path', 'right_path', '90', '-90', '0', '1', 'out_path'])
    assert args.left == 'left_path'
    assert args.right == 'right_path'
    assert args.left_angle == 90
    assert args.right_angle == -90
    assert args.left_camID == 0
    assert args.right_camID == 1
    assert args.output_path == 'out_path'
