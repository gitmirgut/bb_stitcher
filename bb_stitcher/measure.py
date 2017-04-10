"""This module is used to measure nearly planar objects on a surface.

The surface must be parallel to the image plane.
"""
import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker as picker


def get_ratio(image):
    """Determine the ratio to convert from pixel to mm.

    The user must select two points on the image. The selected points, will be used to determine the
    ratio between px and mm.
    Args:
        image (ndarray): Reference image.
    Returns:
         float: Ratio to convert pixel to mm.
    """
    pt_picker = picker.PointPicker()
    points = pt_picker.pick([image], False)
    assert len(points[0]) == 2
    start_point, end_point = points[0]
    distance_mm = float(input('Distance in mm of the two selected points: '))
    ratio = helpers.get_ratio_px_to_mm(start_point, end_point, distance_mm)

    return ratio


def get_origin(image):
    """Determine origin of the image.

    The user must select one point on the image.
    The selected point will be used to define the origin of the image, this will assure that mapped
    coordinates will always be in relation to the origin and not to the image.

    Args:
        image (ndarray): Reference image.

    Returns:
        ndarray: origin (2,)
    """
    pt_picker = picker.PointPicker()
    points = pt_picker.pick([image], False)
    assert len(points[0]) == 1
    return points[0][0]
