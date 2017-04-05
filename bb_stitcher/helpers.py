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
from logging import getLogger
import math

import cv2
import numpy as np

log = getLogger(__name__)


def align_to_display_area(size_left, size_right, homo_left, homo_right):
    """Determine translation matrix & size of two transformed images to align them with display area.

    When two images have been transformed by an homography, it's possible
    that they are not aligned with the displayed area anymore. So they need to
    be translated and the display area must be increased.

    Args:
        size_left (tuple): Size *(width, height)* of the left image.
        size_right (tuple): Size *(width, height)* of the right image.
        homo_left (ndarray): An homography *(3,3)* which is used to transform the left image.
        homo_right (ndarray): An homography *(3,3)* which is used to transform the right image.

    Returns:
        - **homo_trans** (ndarray) -- homography *(3,3)* to translate the left and the right image.
        - **display_size** (tuple) -- Size *(width, height)* of the panorama.
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
    t = [-xmin, -ymin]

    # define translation matrix
    homo_trans = np.array(
        [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    display_size = (math.ceil(xmax - xmin), math.ceil(ymax - ymin))

    return homo_trans, display_size


def add_alpha_channel(image):
    """Add alpha channel to image for transparent areas.

    Args:
        image (ndarray): image of shape *(M,N)* (black/white), *(M,N,3)* (BGR)
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
        width (float): width of the rectangle.
        height (float): width of the rectangle.

    Returns:
        ndarray: rectangle represented by 4 points as ndarray *(4,2)*.
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
        - **harm_rect_a** (ndarray) -- harmonized version of ``rect_a``
        - **harm_rect_b** (ndarray) -- harmonized version of ``rect_b``
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

    This function is the inverted version of ``angles_to_points``.

    Args:
        angle_centers (ndarray): The centers of the ``angles``. *(N,2)*
        points_repr (ndarray): Angles represented by points. *(N,2)*

    Returns:
        ndarray: Angles in rad *(N,)*

    """
    """Calculate angle between vertical line passing through angle_centers and line AB."""
    # https://de.wikipedia.org/wiki/Roll-Nick-Gier-Winkel#/media/File:RPY_angles_of_spaceships_(local_frame).png
    # TODO(zeor_angle) variablen Nullwinkel einbauen, momentan ist entspricht dieser der x-Achse
    assert len(angle_centers) == len(points_repr)

    angles = np.zeros(len(angle_centers), dtype=np.float32)
    for i, angle_center in enumerate(angle_centers):
        point_repr = points_repr[i]
        angle_center_x, angle_center_y = angle_center
        point_repr_x, point_repr_y = point_repr

        # the 0-angle has to be a ray from the center, which is perpendicular to the right border
        # we abstract this ray as a point ``ray_pt`` which always lies on the right side of the
        # center, so ``ray_pt_dis`` has to be just greater 0, we take 10
        ray_pt_dis = 10
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
        r = np.linalg.norm(angle_center - point_repr)

        cos_angle = (p ** 2 + r ** 2 - d ** 2) / (2 * r * p)
        angle = np.arccos(cos_angle)

        if angle_center_y > point_repr_y:
            angle = -angle

        angles[i] = angle

    return angles
