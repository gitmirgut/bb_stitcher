

def add_shared_positional_arguments(parser):
    """Add various shared positional arguments to the `parser` of the :mod:`scripts`.

    Args:
        parser (argparse.ArgumentParser)
    """
    parser.add_argument('left', help='Path of the left image.', type=str)
    parser.add_argument('right', help='Path of the right image.', type=str)
    parser.add_argument('left_angle', help='Rotation angle of the left image', type=int)
    parser.add_argument('right_angle', help='Rotation angle of the right image', type=int)
    parser.add_argument('left_camID', help='Cam ID of the camera which '
                                           'shot the left image.', type=int)
    parser.add_argument('right_camID', help='Cam ID of the camera which '
                                            'shot the right image.', type=int)
    parser.add_argument('out', help='Output path of the stitching data.', type=str)


def add_shared_optional_arguments(parser):
    """Add various shared positional arguments to the `parser` of the :mod:`scripts`.

    Args:
        parser (argparse.ArgumentParser)
    """
    pass
