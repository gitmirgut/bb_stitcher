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
"""This module provides functions to prepare an image or points for stitching."""
import ast
import functools
from logging import getLogger

import cv2
import numpy as np

log = getLogger(__name__)


class Rectificator(object):
    """Class to rectify and remove lens distortion from images and points.

    Attributes:
        initr_m (ndarray): intrinsic matrix of the camera.
        dist_c (ndarray): distortion coefficient of the camera.
    """

    def __init__(self, config):
        """Initialize a rectificator with camera parameters.

        Args:
            config: config file which holds the camera parameters.
        """
        # TODO(gitmirgut) add link to description for loading auto configuration.

        self.intr_m = np.array(ast.literal_eval(config['Rectificator']['INTR_M']))
        self.dist_c = np.array(ast.literal_eval(config['Rectificator']['DIST_C']))
        self.cached_new_cam_mat = None
        self.cached_dim = None
        self.cached_size = None

    def rectify_image(self, image):
        """Remove lens distortion from an image.

        Args:
            image (ndarray): Input (distorted) image.

        Returns:
            ndarray: Output (corrected) image with same size and type as `image`.
        """
        log.info('Start rectification of image with shape {}.'.format(image.shape))
        h, w = image.shape[:2]
        cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(
            self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
        log.debug('new_camera_mat = \n{}'.format(cached_new_cam_mat))
        return cv2.undistort(image, self.intr_m, self.dist_c, None, cached_new_cam_mat)

    def rectify_points(self, points, size):
        """Map points from distorted image to its pos in an undistorted img.

        Args:
            points (ndarray): List of (distorted) points (N,2).
            size (tuple): Size *(width, height)* of the image, which was used for determine the
                                points.

        Returns:
            ndarray: List of (corrected) points.
        """
        points = np.array([points])
        # size = (img_width, img_height)
        log.info(size)

        # size and camera matrix will be cached to speed up rectification if multiple images
        # with same size will be rectified.
        if self.cached_size != size or self.cached_new_cam_mat is None:
            self.size = size
            self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m,
                                                                        self.dist_c,
                                                                        self.size, 1,
                                                                        self.size, 0)
        log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
        return cv2.undistortPoints(
            points, self.intr_m, self.dist_c, None, self.cached_new_cam_mat)[0]


@functools.lru_cache(maxsize=16)
def __get_affine_mat_and_new_size(angle, size=(4000, 3000)):
    """Calculate the affine transformation to rotate image by given angle.

    Args:
        angle (int): Rotation angle in degree. Positive values mean counter-clockwise rotation.
        size (tuple): Size *(width, height)* of the potential image, which was used for determine
                    the points.
    Returns:
        - **affine_mat** (ndarray) -- An affine *(3,3)*--matrix  which rotates and translate image.
        - **new_size** (tuple)  --  Size *(width, height)* of the future image after rotation.
    """
    import math

    # determine the center.
    (width_half, height_half) = tuple(np.array(size) / 2.0)
    center = (width_half - 0.5, height_half - 0.5)

    log.debug('center of the rotation: {}'.format(center))

    # Convert the 3x2 rotation matrix to 3x3 homography.
    rotation_mat = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0), [0, 0, 1]])

    # To get just the rotation.
    rot_matrix_2x2 = rotation_mat[:2, :2]

    # Declare the corners of the image in relation to the center
    corners = np.array([
        [-width_half, height_half],
        [width_half, height_half],
        [-width_half, -height_half],
        [width_half, -height_half]
    ])
    log.debug('corners of the rectangle: {}'.format(corners))

    # get the rotated corners
    corners_rotated = corners.dot(rot_matrix_2x2)
    corners_rotated = np.array(corners_rotated, np.float32)

    # calculate the new dimension of the potential image.
    x_cor = corners_rotated[:, [0][0]]
    right_bound = max(x_cor[x_cor > 0])
    left_bound = min(x_cor[x_cor < 0])
    w = math.ceil(abs(right_bound - left_bound))

    y_cor = corners_rotated[:, [1][0]]
    top_bound = max(y_cor[y_cor > 0])
    bot_bound = min(y_cor[y_cor < 0])
    h = math.ceil(abs(top_bound - bot_bound))

    size_new = (w, h)
    log.debug('size_new = {}'.format(size_new))

    # matrix to center the rotated image
    translation_matrix = np.array([
        [1, 0, math.ceil(w / 2.0 - width_half)],
        [0, 1, math.ceil(h / 2.0 - height_half)],
        [0, 0, 1]
    ])

    # get the affine Matrix
    affine_mat = translation_matrix.dot(rotation_mat)
    log.debug('affine_mat = \n{}'.format(affine_mat))

    return affine_mat, size_new


def rotate_image(image, angle):
    """Rotate image by given angle.

    Args:
        image (ndarray): Input image.
        angle (int): Rotation angle in degree. Positive values mean counter-clockwise rotation.

    Returns:
        - **rot_image** (ndarray) -- Rotated image.
        - **affine_mat** (ndarray) -- An affine *(3,3)*--matrix  which rotates and translate image.
    """
    # TODO(gitmirgut) fix 'one pixel-problem'
    img_size = image.shape[:2][::-1]
    affine_mat, size_new = __get_affine_mat_and_new_size(angle, img_size)
    log.debug("The affine matrix = {} and the new size = {}".format(affine_mat, size_new))
    rot_image = cv2.warpPerspective(image, affine_mat, size_new)
    return rot_image, affine_mat


def rotate_points(points, angle, size):
    """Rotate points by given angle and in relation to the size of an image.

    Args:
        points (ndarray): List of points (N, 2).
        angle (int): Rotation angle in degree. Positive values mean counter-clockwise rotation.
        size (tuple): Size *(width, height)* of the image, which was used for determine the
                    points.

    Returns:
        ndarray: Rotated points (N, 2).
    """
    points = np.array([points])
    log.debug('Start rotate points.')
    affine_mat, __ = __get_affine_mat_and_new_size(angle, size)
    return cv2.transform(points, affine_mat[0:2])[0]
