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


@functools.lru_cache(maxsize=16)
def __wrapper_getOptimalNewCameraMatrix(intr_m, dist_c, size):
    """This wrapper is just for speed up."""
    new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(intr_m, dist_c, size, 1, size, 0)
    return new_cam_mat


class Rectificator(object):
    """Class to rectify and remove lens distortion from images and points.

    Attributes:
        initr_m (ndarray): intrinsic matrix.
        dist_c (ndarray): distortion coefficient.
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

    def rectify_image(self, img):
        """Remove Lens distortion from an image.

        Args:
            img (ndarray): Input (distorted) image.

        Returns:
            ndarray: Output (corrected) image with same size and type as `img`.
        """
        log.info('Start rectification of image with shape {}.'.format(img.shape))
        h, w = img.shape[:2]
        cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(
            self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
        log.debug('new_camera_mat = \n{}'.format(cached_new_cam_mat))
        return cv2.undistort(img, self.intr_m, self.dist_c, None, cached_new_cam_mat)

    def rectify_points(self, points, img_width, img_height):
        """Map points from distorted image to its pos in an undistorted img.

        Args:
            points (ndarray): List of (distorted) points (N,2).
            img_width (int): The width of the original image, which was used for determine the
                                points.
            img_height (int): The height of the original image, which was used for determine
                                the points.

        Returns:
            ndarray: List of (corrected) points.
        """
        points = np.array([points])
        size = (img_width, img_height)
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
def __get_affine_mat_and_new_size(angle, img_width, img_height):
    """Calculate the affine transformation to rotate image by given angle.

    Args:
        angle (int): angle in degree.
        img_width (int): The width of the original image, which was used for determine the
                            points.
        img_height (int): The height of the original image, which was used for determine
                            the points.
    Returns:
        - **affine_mat** (ndarray) -- An affine *(3,3)*--matrix  which rotates image .
        - **new_size** (tuple)  --  Size *(width, height)* of the future image after rotation .
    """
    # Get img size
    size = (img_width, img_height)
    center = tuple(np.array(size) / 2.0)
    (width_half, height_half) = center

    # Convert the 3x2 rotation matrix to 3x3 ''homography''
    rotation_mat = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0), [0, 0, 1]])

    # To get just the rotation
    rot_matrix_2x2 = rotation_mat[:2, :2]

    # Declare the corners of the image in relation to the center
    corners = np.array([
        [-width_half, height_half],
        [width_half, height_half],
        [-width_half, -height_half],
        [width_half, -height_half]
    ])

    # get the rotated corners
    corners_rotated = corners.dot(rot_matrix_2x2)
    corners_rotated = np.array(corners_rotated, np.float32)

    # get the rectangle which would surround the rotated image
    __, __, w, h = cv2.boundingRect(np.array(corners_rotated))

    # boundingRect is 1px bigger so remove it
    size_new = (w - 1, h - 1)
    log.debug('size_new = {}'.format(size_new))

    # matrix to center the rotated image
    translation_matrix = np.array([
        [1, 0, int(w / 2 - width_half)],
        [0, 1, int(h / 2 - height_half)],
        [0, 0, 1]
    ])

    # get the affine Matrix
    affine_mat = translation_matrix.dot(rotation_mat)
    log.debug('affine_mat = \n{}'.format(affine_mat))

    return affine_mat, size_new

    def rotate_image(self):
        pass


if __name__ == '__main__':
    import time

    start = time.time()
    mat, size = __get_affine_mat_and_new_size(90, 4000, 3000)
    print(mat)
    print(mat.shape)
    end = time.time()
    print(end-start)