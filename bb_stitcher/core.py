"""Module to connect stitcher and mapping from image coordinates to world coordinates."""
import cv2
import numpy as np

import bb_stitcher.measure as measure
import bb_stitcher.stitcher as stitcher


class Surveyor(object):
    """Class to determine the relationship between two images of one comb side .

    The ``Surveyor`` determines all needed data to stitch two images from different areas of one
    comb side to a complete view of the comb. On this basis the ``Surveyor`` can be used to map the
    coordinates from these images to hive coordinates.
    """

    def __init__(self, config):
        """Initialize Surveyor."""
        self.config = config
        self.stitching_params = None
        self.origin = None
        self.ratio_px_mm = None
        self.world_homo = None
        self.cam_id_l = None
        self.cam_id_r = None

    def determine_mapping_parameters(self, path_l, path_r, angl_l, angl_r,
                                     cam_id_l, cam_id_r, stitcher_type):
        """Determine the parameters to mapping parameters.

        This functions is used to calculate all needed data to stitch two images and to map
        image coordinates and angels to hive coordinates.

        Args:
            path_l (str): Path to the left image.
            path_r (str): Path to the right image.
            angl_l (int): Angle in degree to rotate left image.
            angl_r (int): Angle in degree to rotate right image.
            cam_id_l (int): ID of the camera, which shot the left image.
            cam_id_r (int): ID of the camera, which shot the right image.
            stitcher_type (Stitcher): Stitcher to use for stitching of the images.
        """
        assert stitcher.Stitcher in stitcher_type.__bases__
        img_l = cv2.imread(path_l, -1)
        img_r = cv2.imread(path_r, -1)
        stitch = stitcher_type(self.config)
        stitch.estimate_transform(img_l, img_r, angl_l, angl_r)

        panorama = stitch.compose_panorama(img_l, img_r)

        self.stitching_params = stitch.get_parameters()
        self.origin = measure.get_origin(panorama)
        self.ratio_px_mm = measure.get_ratio(panorama)
        trans_homo = np.array([
            [1, 0, -self.origin[0]],
            [0, 1, -self.origin[1]],
            [0, 0, 1]], dtype=np.float64)
        ratio_homo = np.array([
            [self.ratio_px_mm, 0, 0],
            [0, self.ratio_px_mm, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        self.world_homo = ratio_homo.dot(trans_homo)
        self.cam_id_l = cam_id_l
        self.cam_id_r = cam_id_r

    def map_points_angles(self, points, angles, cam_id):
        u"""Map image points and angles to points and angles in relation to world/hive.

        This happens under the assumption that the mapping parameters were estimated or loaded
        before.

        Args:
            points (ndarray): List of points from left image in px *(N,2)*.
            angles (ndarray): Angles in rad (length *(N,)*).
            cam_id (ndarray): ID of the camera, which shot the image.

        Returns:
            - **points_mapped** (ndarray) -- ``points`` mapped to hive in mm *(N,2)*.
            - **angles_mapped** (ndarray) -- ``angles`` mapped to  *(N,)*.

        Note:
            For all angles in ``angles`` it is assumed that a 0°-angle shows to the right border of
            the image and that a positive angle means clockwise rotation.
        """
        homo_left = self.stitching_params.homo_left
        homo_right = self.stitching_params.homo_right
        size_left = self.stitching_params.size_left
        size_right = self.stitching_params.size_right
        pano_size = self.stitching_params.pano_size
        stitch = stitcher.Stitcher(self.config)

