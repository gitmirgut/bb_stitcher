"""
This module is used to measure nearly planar objects on a surface, which is parallel to the image
plane.
"""
import numpy as np

import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker as picker


def calc_ratio(image):
    """Determine the ratio to convert from pixel to mm.
    
    The user must select two points on the image. The selected points, will be used to determine the
    ratio between px and mm.
    Args:
        image (ndarray): Reference image.
    Returns:
         flaot: Ratio to convert pixel to mm.
    """
    pt_picker = picker.PointPicker()
    points = pt_picker.pick([image], False)
    assert len(points[0]) == 2
    start_point, end_point = points[0]
    print(start_point)
    print(end_point)
    distance_mm = float(input('Distance in mm of the two selected points: '))
    ratio = helpers.get_ratio_px_to_mm(start_point, end_point, distance_mm)

    return ratio
