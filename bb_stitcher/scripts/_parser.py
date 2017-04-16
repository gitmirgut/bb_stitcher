"""This module provides argument parsers for the scripts."""
import argparse
import os
import textwrap

import bb_stitcher.io_utils as io_utils


def filepath(string):
    """Define a special argument type for the argument parser.

    It checks if the given string is a valid file path.
    """
    if not os.path.exists(string):
        msg = 'File "{path}" does not exists.'.format(path=string)
        raise argparse.ArgumentTypeError(msg)
    return string


def output_path(string):
    """Define a special argument type for the argument parser.

    It checks if the output path has a valid extension.
    """
    __, ext = os.path.splitext(string)
    if ext not in io_utils.valid_ext:
        msg = 'Extension "{ext}" is not a valid extension, please use ' \
              'one of these {valid_ext} extensions.'.format(ext=ext,
                                                            valid_ext=','.join(io_utils.valid_ext))
        raise argparse.ArgumentTypeError(msg)
    return string


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
                                 help='Path of the left image.', type=filepath)

    estimate_parser.add_argument('right',
                                 help='Path of the right image.', type=filepath)

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
                                 '''.format(ext=','.join(io_utils.valid_ext))), type=output_path)

    # Define composer parser ----------------------------------------------------------------------
    compose_parser.add_argument('left', help='Path of the left image.', type=filepath)
    compose_parser.add_argument('right', help='Path of the right image.', type=filepath)
    return main_parser
