"""This is the main script and entry point for the bb_stitcher."""

import bb_stitcher.core as core
import bb_stitcher.helpers as helpers
import bb_stitcher.scripts._parser as _parser
import bb_stitcher.stitcher as stitcher


def estimate_params(args):
    """Interpret and execute the arguments from the arguments parser."""
    surveyor = core.Surveyor(helpers.get_default_config())
    surveyor.determine_mapping_parameters(args.left, args.right,
                                          args.left_angle, args.right_angle,
                                          args.left_camID, args.right_camID,
                                          stitcher.RectangleStitcher)
    surveyor.compose_panorama(args.left, args.right)
    surveyor.save(args.out)


def compose_params(args):
    pass


def main():
    """Parse the arguments of parser."""
    main_parser = _parser.get_parser()
    args = main_parser.parse_args()
    estimate_params(args)


if __name__ == '__main__':
    main()
