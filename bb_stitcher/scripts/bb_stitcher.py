"""This is the main script and entry point for the bb_stitcher."""
import argparse
import os
import textwrap

import cv2

import bb_stitcher.core as core
import bb_stitcher.helpers as helpers
import bb_stitcher.io_utils as io_utils
import bb_stitcher.stitcher as stitcher

"""
Define the subcommands to work with ArgumentParser.
"""


def estimate_params(args):
    """Execute the subcommand 'estimate' from the parser."""
    surveyor = core.Surveyor(helpers.get_default_config())
    surveyor.determine_mapping_parameters(args.left, args.right,
                                          args.left_angle, args.right_angle,
                                          args.left_camID, args.right_camID,
                                          stitcher.RectangleStitcher)
    surveyor.compose_panorama(args.left, args.right)
    surveyor.save(args.out)


def compose_params(args):
    """Execute the subcommand 'compose' from the parser."""
    surveyor = core.Surveyor(helpers.get_default_config())
    surveyor.load(args.data)
    img = surveyor.compose_panorama(args.left, args.right)
    cv2.imwrite(args.out, img)
    pass


"""
Define special argument types for the argument parser.
"""


def _exist_path(string):
    """Check if the given string is a valid file string."""
    if not os.path.exists(string):
        msg = 'File "{path}" does not exists.'.format(path=string)
        raise argparse.ArgumentTypeError(msg)
    return string


def _data_path(string):
    """Check if the path has a valid extension."""
    __, ext = os.path.splitext(string)
    if ext not in io_utils.valid_ext:
        msg = 'The Extension "{ext}" of "{path}" is not a valid extension, please use ' \
              'one of these {valid_ext} extensions.'.format(ext=ext,
                                                            path=string,
                                                            valid_ext=','.join(io_utils.valid_ext))
        raise argparse.ArgumentTypeError(msg)
    return string


def _img_path(string):
    """Check if the path is a valid image."""
    valid_ext = ['.jpeg', '.jpg', '.png']
    __, ext = os.path.splitext(string)
    if ext not in valid_ext:
        msg = 'Did not understand the image type of {path}, please use one of these {valid_ext}' \
              ' extensions.'.format(path=string,
                                    valid_ext=','.join(valid_ext))
        raise argparse.ArgumentTypeError(msg)
    return string


def _stitching_data(string):
    """Check if the path is a valid data file, which holds params."""
    string = _data_path(string)
    return _exist_path(string)


"""
Define the main parser.
"""


def _get_main_parser():
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

    estimate_parser.add_argument('left', help='Path of the left image.', type=_exist_path)
    estimate_parser.add_argument('right', help='Path of the right image.', type=_exist_path)

    estimate_parser.add_argument('left_angle', help='Rotation angle of the left image '
                                                    '(counter-clockwise).', type=int)
    estimate_parser.add_argument('right_angle', help='Rotation angle of the right image '
                                                     '(counter-clockwise).', type=int)

    estimate_parser.add_argument('left_camID',
                                 help='Cam ID of the camera which shot the left image.', type=int)
    estimate_parser.add_argument('right_camID',
                                 help='Cam ID of the camera which shot the right image.', type=int)

    estimate_parser.add_argument('out',
                                 help=textwrap.dedent('''\
                                         Output path of the stitching data.
                                         Supported Types: {ext}
                                         '''.format(ext=','.join(io_utils.valid_ext))),
                                 type=_data_path)
    estimate_parser.set_defaults(func=estimate_params)

    # Define composer parser ----------------------------------------------------------------------
    compose_parser.add_argument('left', help='Path of the left image.', type=_exist_path)
    compose_parser.add_argument('right', help='Path of the right image.', type=_exist_path)

    compose_parser.add_argument('data', help='Path of the file which holds the stitching data.',
                                type=_stitching_data)
    compose_parser.add_argument('out', help='Output path of the stitchted images.', type=_img_path)
    compose_parser.set_defaults(func=compose_params)
    return main_parser


def main():
    """Parse the arguments of parser."""
    # define the main parser
    main_parser = _get_main_parser()
    args = main_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
