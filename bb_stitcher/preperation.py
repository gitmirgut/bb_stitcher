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
from logging import getLogger

import cv2
import numpy as np

log = getLogger(__name__)


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

    def rectify_points(self, points, img_height, img_width):
        """Map points from distorted image to its pos in an undistorted img.

        Args:
            points (ndarray (N,2)): List of (distorted) points.
            img_height (int): The height of the original image, which was used for determine
                                the points.
            img_width (int): The width of the original image, which was used for determine the
                                points.

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
