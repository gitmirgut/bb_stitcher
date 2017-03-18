import ast
import configparser
import cv2
from logging import getLogger
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
        #TODO(LINK) add link to discription for loading autoconfiguration.

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
        cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m, self.dist_c, (w, h), 1, (w, h), 0)
        log.debug('new_camera_mat = \n{}'.format(cached_new_cam_mat))
        return cv2.undistort(img, self.intr_m, self.dist_c, None, cached_new_cam_mat)

    def rectify_points(self, points, img_height, img_width):
        """Map points from distorted image to its pos in an undistorted img.

            Args:
                points (ndarray): List of (distorted) points.
                img_height (int): The height of the original image, which was used for determine the points.
                img_width (int): The width of the original image, which was used for determine the points.

            Returns:
                ndarray: List of (corrected) points.
        """
        points = np.array([points])
        size = (img_width, img_height)
        log.info(size)
        if self.cached_size != size or self.cached_new_cam_mat is None:
            self.size = size
            self.cached_new_cam_mat, __ = cv2.getOptimalNewCameraMatrix(self.intr_m,
                                                                        self.dist_c,
                                                                        self.size, 1,
                                                                        self.size, 0)
        log.debug('new_camera_mat = \n{}'.format(self.cached_new_cam_mat))
        return cv2.undistortPoints(points, self.intr_m, self.dist_c, None, self.cached_new_cam_mat)[0]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('default_config.ini')
    a = Rectificator(config)
