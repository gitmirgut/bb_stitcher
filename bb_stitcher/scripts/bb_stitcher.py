"""This is the main script and entry point for the bb_stitcher."""
import cv2

import bb_stitcher.core as core
import bb_stitcher.helpers as helpers
import bb_stitcher.scripts._parser as _parser
import bb_stitcher.stitcher as stitcher


def process_images(args):
    """Interpret and execute the arguments from the arguments parser."""
    surveyor = core.Surveyor(helpers.get_default_config())
    surveyor.determine_mapping_parameters(args.left, args.right,
                                          args.left_angle, args.right_angle,
                                          args.left_camID, args.right_camID,
                                          stitcher.RectangleStitcher)
    pano = surveyor.compose_panorama(args.left, args.right)
    cv2.imwrite(args.out, pano)
    pass


def main():
    """Parse the arguments of parser."""
    main_parser = _parser.get_parser()
    args = main_parser.parse_args()
    process_images(args)


if __name__ == '__main__':
    main()
