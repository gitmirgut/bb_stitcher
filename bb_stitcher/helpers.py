#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.
"""This module provides various helper functions."""
import collections
import configparser
from logging import getLogger
import os

import cv2
import numpy as np

log = getLogger(__name__)


def get_default_config():
    """Return the default config."""
    default_config = configparser.ConfigParser()
    path_config = os.path.join(os.path.dirname(__file__), 'default_config.ini')
    default_config.read(path_config)
    return default_config


def get_default_debug_config():
    """Return the default logging config file."""
    default_config = configparser.ConfigParser()
    path_config = os.path.join(os.path.dirname(__file__), 'logging_config.ini')
    default_config.read(path_config)
    return default_config


def get_boundaries(size_left, size_right, homo_left, homo_right):
    """Determine the boundaries of two transformed images.

    When two images have been transformed by homographies to a 'shared space' (which holds both
    images), it's possible that this 'shared space' is not aligned with the displayed area.
    Its possible that various points are outside of the display area.
    This function determines the max/min values of x and y of the both images in shared space
    in relation to the origin of the display area.

    Example:
        .. code::

                         *--------*    *--------*
                         |        |    |        |
                         |  left  |    | right  |
                         |        |    |        |
                         *--------*    *--------*
                             \            /
                   homo_left  \          / homo_right
                               \        /
                                v      v

            shared space:      *--------*
                  +~~~~~~~~~~~~|        |~~~~+
                  ;  *--------*| right  |    ;
                  ;  |        ||        |    ;
                  ;  |  left  |*--------*    ;
                  ;  |        |              ;
                  ;  *--------* display_area ;
                  +~~~~~~~~~~~~~~~~~~~~~~~~~~+

    (In this example ``xmin`` would be the x value of the left border from the left image and
    ``ymin`` would be the y value of the top border from the right image)

    Args:
        size_left (tuple): Size *(width, height)* of the left image.
        size_right (tuple): Size *(width, height)* of the right image.
        homo_left (ndarray): An homography *(3,3)* which is used to transform the left image.
        homo_right (ndarray): An homography *(3,3)* which is used to transform the right image.

    Returns:
        -- **xmin** (float) -- Minimal x value of both images after transformation.
        -- **ymin** (float) -- Minimal y value of both images after transformation.
        -- **xmax** (float) -- Maximal x value of both images after transformation.
        -- **ymax** (float) -- Maximal x value of both images after transformation.
    """
    h_l, w_l = size_left
    h_r, w_r = size_right

    corners_l = np.float32([
        [0, 0],
        [0, w_l],
        [h_l, w_l],
        [h_l, 0]
    ]).reshape(-1, 1, 2)
    corners_r = np.float32([
        [0, 0],
        [0, w_r],
        [h_r, w_r],
        [h_r, 0]
    ]).reshape(-1, 1, 2)

    # transform the corners of the images, to get the dimension of the
    # transformed images and stitched image
    corners_tr_l = cv2.perspectiveTransform(corners_l, homo_left)
    corners_tr_r = cv2.perspectiveTransform(corners_r, homo_right)

    pts = np.concatenate((corners_tr_l, corners_tr_r), axis=0)
    # measure the max values in x and y direction to get the translation vector
    # so that whole image will be shown
    [xmin, ymin] = np.float32(pts.min(axis=0).ravel())
    [xmax, ymax] = np.float32(pts.max(axis=0).ravel())

    Bounderies = collections.namedtuple('Bounderies', ['xmin', 'ymin', 'xmax', 'ymax'])
    return Bounderies(xmin, ymin, xmax, ymax)


def get_transform_to_origin_mat(xmin, ymin):
    """Determine homography matrix to align 'shared_space' to display area origin.

    Example:
        .. code::

            shared space:      *--------*
                  +~~~~~~~~~~~~|        |~~~~+
                  ;  *--------*| right  |    ;
                  ;  |        ||        |    ;
                  ;  |  left  |*--------*    ;
                  ;  |        |              ;
                  ;  *--------* display_area ;
                  +~~~~~~~~~~~~~~~~~~~~~~~~~~+

                                |
                                | transformation to origin
                                V

                  +~~~~~~~~~*--------*~~~~~~+
                  ;         |        |      ;
                  *--------*| right  |      ;
                  |        ||        |      ;
                  |  left  |*--------*      ;
                  |        |                ;
                  *--------* display_area   ;
                  ;                         ;
                  +~~~~~~~~~~~~~~~~~~~~~~~~~+
    Args:
        xmin (float): Minimal x value of images in 'shared space'.
        ymin (float): Minimal y value of images in 'shared space'.

    Returns:
        ndarray: *(3,3)* homography to align 'shared space' the to the origin of display area.

    See Also:
        - :meth:`get_boundaries`
    """
    t = [-xmin, -ymin]

    # define translation matrix
    homo_trans = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)  # translate
    return homo_trans


def add_alpha_channel(image):
    """Add alpha channel to image for transparent areas.

    Args:
        image (ndarray): Image of shape *(M,N)* (black/white), *(M,N,3)* (BGR)
                        or *(M,N,4)* already with alpha channel.

    Returns:
        ndarray: ``image`` extended with alpha channel

    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            return image
        else:
            raise Exception('Shape {} of image is unknown cannot add alpha channel. Valid image'
                            'shapes are (N,M), (N,M,3), (N,M,4).'.format(str(image.shape)))
    else:
        raise Exception('Shape {} of image is unknown cannot add alpha channel. Valid image shapes'
                        'are (N,M), (N,M,3), (N,M,4).'.format(str(image.shape)))


def form_rectangle(width, height):
    """Return a rectangle represented by 4 points ndarray *(4,2)*.

    The starting point is the Origin and the points are sorted in clockwise order.

    Args:
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.

    Returns:
        ndarray: Rectangle represented by 4 points as ndarray *(4,2)*.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = 0, 0
    rect[1] = width, 0
    rect[2] = width, height
    rect[3] = 0, height

    return rect


def sort_pts(points):
    r"""Sort points as convex quadrilateral.

    Sort points in clockwise order, so that they form a convex quadrilateral.

    Example:
        .. code::

            pts:                sorted_pts:
                 x   x                      A---B
                              --->         /     \
               x       x                  D-------C

    Args:
        points (ndarray): Array of points *(N,2)*.

    Returns:
        ndarray: Clockwise ordered ``points`` *(N,2)*, where the most up left point is the \
        starting point.
    """
    assert (len(points) == 4)

    # calculate the barycentre / centre of gravity
    barycentre = points.sum(axis=0) / 4

    # var for saving the points in relation to the barycentre
    bary_vectors = np.zeros((4, 2), np.float32)

    # var for saving the A point of the origin
    A = None
    min_dist = None

    for i, point in enumerate(points):

        # determine the distance to the origin
        cur_dist_origin = np.linalg.norm(point)

        # save the A point of the origin
        if A is None or cur_dist_origin < min_dist:
            min_dist = cur_dist_origin
            A = i

        # determine point in relation to the barycentre
        bary_vectors[i] = point - barycentre

    angles = np.zeros(4, np.float32)
    # determine the angles of the different points in relation to the line
    # between closest point of origin (A) and barycentre
    for i, bary_vector in enumerate(bary_vectors):
        if i != A:
            cur_angle = np.arctan2(
                (np.linalg.det((bary_vectors[A], bary_vector))), np.dot(
                    bary_vectors[A], bary_vector))
            if cur_angle < 0:
                cur_angle += 2 * np.pi
            angles[i] = cur_angle
    index_sorted = np.argsort(angles)
    sorted_pts = np.zeros((len(points), 2), np.float32)
    for i in range(len(points)):
        sorted_pts[i] = points[index_sorted[i]]
    return sorted_pts


def raw_estimate_rect(points):
    """Abstract an rectangle from an convex quadrilateral.

    The convex quadrilateral is defined by ``Points``. The points must be sorted in clockwise order
    where the most up left point is the starting point. (see sort_pts)

    Example:
        .. code::

            points:             rectangled points:
                 A---B                    A'------B'
                /     \       --->        |       |
               D-------C                  D'------C'

    The dimension of the rectangle is estimated in the following manner:
    ``|A'B'|=|D'C'|=max(|AB|,|DC|)`` and ``|A'D'|=|B'C'|=max(|AD|,|BC|)``

    Args:
        points (ndarray): Array of clockwise ordered points *(4,2)*, where most up left point is\
        the starting point.

    Returns:
        ndarray: 'Rectangled' points (the rectangle is aligned to the origin).
    """
    # TODO(gitmirgut) add link to sort_pts
    A = points[0]
    B = points[1]
    C = points[2]
    D = points[3]

    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    CD = np.linalg.norm(D - C)
    DA = np.linalg.norm(D - A)

    hori_len = max(AB, CD)
    vert_len = max(BC, DA)

    dest_rect = form_rectangle(hori_len, vert_len)

    return dest_rect


def harmonize_rects(rect_a, rect_b):
    """Harmonize two rectangles in their vertical dimension.

    Example:
        .. code::

           rect_a:    rect_b:        harm_rect_a:       harm_rect_b:

                        W-----X         A'--------------B'    W'----X'
            A-----B     |     |         |               |     |     |
            |     |     |     |   -->   |               |     |     |
            D-----C     |     |         |               |     |     |
                        Z-----Y         D'--------------C'    Z'----Y'

    Args:
        rect_a (ndarray): Array of clockwise ordered points *(4,2)*, where most up left point is\
        the starting point.
        rect_b (ndarray): Same as ``rect_a``

    Returns:
        - **harm_rect_a** (ndarray) -- Harmonized version of ``rect_a``
        - **harm_rect_b** (ndarray) -- Harmonized version of ``rect_b``
    """
    A = rect_a[0]
    B = rect_a[1]
    C = rect_a[2]
    D = rect_a[3]

    W = rect_b[0]
    X = rect_b[1]
    Y = rect_b[2]
    Z = rect_b[3]

    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    CD = np.linalg.norm(D - C)
    DA = np.linalg.norm(D - A)

    assert AB == CD and BC == DA

    WX = np.linalg.norm(X - W)
    XY = np.linalg.norm(Y - X)
    YZ = np.linalg.norm(Z - Y)
    ZW = np.linalg.norm(W - Z)

    assert WX == YZ and XY == ZW

    hori_a = AB
    vert_a = BC

    hori_b = WX
    vert_b = XY

    if vert_a > vert_b:
        harm_vert_b = vert_a
        ratio = vert_a / vert_b
        harm_hori_b = ratio * hori_b
        harm_rect_b = form_rectangle(harm_hori_b, harm_vert_b)
        return rect_a, harm_rect_b
    else:
        harm_vert_a = vert_b
        ratio = vert_b / vert_a
        harm_hori_a = ratio * hori_a
        harm_rect_a = form_rectangle(harm_hori_a, harm_vert_a)
        return harm_rect_a, rect_b


def angles_to_points(angle_centers, angles, distance=22):
    r"""Calculate point representations of angles.

    The angle point representations ``points_reprs`` are calculated in  dependency of the
    ``angle_center`` and the ray starting from this center, which is perpendicular to the right
    border. Positive angles will be interpreted as clockwise rotation.

    Example:
        .. code::

            angle_center
                  *--------x-Axis------>
                   \         |
                    \ angle /
                     \     /
                      \ --´
                       \
            points_repr *
                         \
                          v

    Args:
        angle_centers (ndarray): The centers of the ``angles``. *(N,2)*
        angles (ndarray): Angles in rad (length *(N,)*).
        distance (int): The distance between the ``angle_centers`` and the point representations.

    Returns:
        - **points_repr** (ndarray) -- Angles represented by points. *(N,2)*

    See Also:
        - :meth:`points_to_angles`
    """
    assert len(angle_centers) == len(angles)
    points_repr = np.zeros((len(angle_centers), 2), dtype=np.float32)
    for i, center in enumerate(angle_centers):
        z_rotation = np.array(angles[i])
        # remove round
        points_repr[i, 0] = center[0] + distance * np.cos(z_rotation)
        points_repr[i, 1] = center[1] + distance * np.sin(z_rotation)
    return points_repr


def points_to_angles(angle_centers, points_repr):
    """Convert angle point representation back to normal angle.

    This function is the inverted version of :meth:`angles_to_points`.

    Args:
        angle_centers (ndarray): The centers of the ``angles``. *(N,2)*
        points_repr (ndarray): Angles represented by points. *(N,2)*

    Returns:
        ndarray: Angles in rad *(N,)*

    See Also:
        - :meth:`angles_to_points`
    """
    """Calculate angle between vertical line passing through angle_centers and line AB."""
    # https://de.wikipedia.org/wiki/Roll-Nick-Gier-Winkel#/media/File:RPY_angles_of_spaceships_(local_frame).png
    # TODO(zeor_angle) variablen Nullwinkel einbauen, momentan ist entspricht dieser der x-Achse
    import warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    assert len(angle_centers) == len(points_repr)

    angles = np.zeros(len(angle_centers), dtype=np.float32)
    for i, angle_center in enumerate(angle_centers):
        point_repr = points_repr[i]
        angle_center_x, angle_center_y = angle_center
        point_repr_x, point_repr_y = point_repr

        # the 0-angle has to be a ray from the center, which is perpendicular to the right border
        # we abstract this ray as a point ``ray_pt`` which always lies on the right side of the
        # center, so ``ray_pt_dis`` has to be just greater 0, we take 80
        ray_pt_dis = np.float64(80)
        ray_pt = np.array([angle_center_x + ray_pt_dis, angle_center_y])

        """
        angle_center      p       ray_pt
                  *---------------*
                   \         |   /
                    \ angle /   /
                 r   \     /   /  d
                      \ --´   /
                       \     /
                        \   /
                         \ /
               point_repr *
        """

        d = np.linalg.norm(ray_pt - point_repr)
        p = ray_pt_dis
        r = np.linalg.norm(np.float64(angle_center) - point_repr)
        if r == 0:
            raise Exception('Angle center point {} and angle point representation {}'
                            ' seams to be the same.'.format(angle_center, point_repr))
        cos_angle = (p ** 2 + r ** 2 - d ** 2) / (2 * r * p)

        # this is due to some arithmetic problems, where cos_angle is something like
        # cos_angle = 1.000000000000008 which leads to an error.
        if cos_angle > 1 and np.isclose(cos_angle, 1, atol=1e-10, rtol=1e-9):
            cos_angle = 1
        elif cos_angle < -1 and np.isclose(cos_angle, -1, atol=1e-10, rtol=1e-9):
            cos_angle = -1
        try:
            angle = np.arccos(cos_angle)
        except Warning:
            print('angle_centers = {angle_centers}, '
                  'points_repr = {points_repr}'.format(angle_centers=angle_centers,
                                                       points_repr=points_repr))
            raise Exception('arccos can not handle {cos_angle}'.format(cos_angle=cos_angle))

        if angle_center_y > point_repr_y:
            angle = -angle

        angles[i] = angle

    return angles


def get_ratio_px_to_mm(start_point, end_point, distance_mm):
    """Return ratio between pixel and millimetre.

    The function calculates the distance of two points (``start_point``, ``end_point``) in pixels
    and then calculates ratio using the distance in pixels and the distance in mm ``distance_mm``.

    Args:
        start_point (ndarray): Start point of the reference Line Segment *(2,)*
        end_point (ndarray): End point of the reference Line Segment *(2,)*
        distance_mm (float): The distance between the ``start_point`` and ``end_point`` of the \
        line segment in real world in mm.

    Returns:
        float: The ratio between px and mm (the length of 1px in mm).

    """
    distance_px = np.linalg.norm(end_point - start_point)
    return distance_mm / distance_px
