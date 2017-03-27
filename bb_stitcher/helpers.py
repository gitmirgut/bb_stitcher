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
        - **homo_trans** (ndarray) -- An homography *(3,3)* to translate the left and the right
                                    image so that is they will be aligned with the display area.
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
        ndarray: image with alpha channel

    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] == 4:
            return image
        else:
            raise Exception('Shape {} of image is unknown cannot add alpha channel. '
                            'Valid image shapes are (N,M), (N,M,3), (N,M,4).'.format(str(image.shape)))
    else:
        raise Exception('Shape {} of image is unknown cannot add alpha channel. Valid image shapes'
                        'are (N,M), (N,M,3), (N,M,4).'.format(str(image.shape)))


if __name__ == '__main__':
    pass
