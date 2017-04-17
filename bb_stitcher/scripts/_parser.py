"""This module provides argument parsers for the scripts."""
import argparse
import os
import textwrap

import bb_stitcher.io_utils as io_utils


def exist_path(string):
    """Define a special argument type for the argument parser.

    It checks if the given string is a valid file string.
    """
    if not os.path.exists(string):
        msg = 'File "{path}" does not exists.'.format(path=string)
        raise argparse.ArgumentTypeError(msg)
    return string


def data_path(string):
    """Define a special argument type for the argument parser.

    It checks if the path has a valid extension.
    """
    __, ext = os.path.splitext(string)
    if ext not in io_utils.valid_ext:
        msg = 'The Extension "{ext}" of "{path}" is not a valid extension, please use ' \
              'one of these {valid_ext} extensions.'.format(ext=ext,
                                                            path=string,
                                                            valid_ext=','.join(io_utils.valid_ext))
        raise argparse.ArgumentTypeError(msg)
    return string


def img_path(string):
    """Define a special argument type for the argument parser.

    It checks if the path is a valid image.
    """
    valid_ext = ['.jpeg', '.jpg', '.png']
    __, ext = os.path.splitext(string)
    if ext not in valid_ext:
        msg = 'Did not understand the image type of {path}, please use one of these {valid_ext}' \
              ' extensions.'.format(path=string,
                                    valid_ext=','.join(valid_ext))
        raise argparse.ArgumentTypeError(msg)
    return string


def stitching_data(string):
    string = data_path(string)
    return exist_path(string)


def get_parser():
    """Return the main parser for the :mod:`.bb_stitcher` script."""
    main_parser = argparse.ArgumentParser(
        prog='bb_stitcher',
        usage='%(prog)s <command> [options]',
        description='This will stitch two images and return the needed data for reproducing the'
                    'stitching with points and angles.',
    )
    subparsers = main_parser.add_subparsers(title="Commands")

    estimate_parser = subparsers.add_parser('estimate',
                                            aliases=['est'], usage='bb_stitcher estimate',
                                            help='Estimate stitching parameters.',
                                            formatter_class=argparse.RawTextHelpFormatter)
    compose_parser = subparsers.add_parser('compose',
                                           aliases=['com'], help='Compose panorama.')

    # Define estimation parser --------------------------------------------------------------------
    estimate_parser.add_argument('type',
                                 choices=['fb', 'rect'], type=str,
                                 help=textwrap.dedent('''\
                                 Define the stitcher to use:
                                    fb - FeatureBasedStitcher
                                    rect - RectangleStitcher
                                '''))

    estimate_parser.add_argument('left',
                                 help='Path of the left image.', type=exist_path)

    estimate_parser.add_argument('right',
                                 help='Path of the right image.', type=exist_path)

    estimate_parser.add_argument('left_angle',
                                 help='Rotation angle of the left image '
                                      '(counter-clockwise).', type=int)

    estimate_parser.add_argument('right_angle',
                                 help='Rotation angle of the right image '
                                      '(counter-clockwise).', type=int)

    estimate_parser.add_argument('left_camID',
                                 help='Cam ID of the camera which shot the left image.', type=int)

    estimate_parser.add_argument('right_camID',
                                 help='Cam ID of the camera which shot the right image.', type=int)

    estimate_parser.add_argument('out',
                                 help=textwrap.dedent('''\
                                 Output path of the stitching data.
                                 Supported Types: {ext}
                                 '''.format(ext=','.join(io_utils.valid_ext))), type=data_path)

    # Define composer parser ----------------------------------------------------------------------
    compose_parser.add_argument('left', help='Path of the left image.', type=exist_path)
    compose_parser.add_argument('right', help='Path of the right image.', type=exist_path)
    compose_parser.add_argument('data', help='Path of the file which holds the stitching data.',
                                type=stitching_data)
    compose_parser.add_argument('out', help='Output path of the stitchted images.', type=img_path)
    return main_parser
