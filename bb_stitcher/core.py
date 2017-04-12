"""Module to connect stitcher and mapping from image coordinates to world coordinates."""
import collections

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
        self.homo_left = None
        self.homo_right = None
        self.size_left = None
        self.size_right = None
        self.pano_size = None
        self.origin = None
        self.ratio_px_mm = None
        self.world_homo = None
        self.cam_id_l = None
        self.cam_id_r = None

        # these will not be exported
        self._world_homo_left = None
        self._world_homo_right = None

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

        stitching_params = stitch.get_parameters()
        self.homo_left = stitching_params.homo_left
        self.homo_right = stitching_params.homo_right
        self.size_left = stitching_params.size_left
        self.size_right = stitching_params.size_right
        self.pano_size = stitching_params.pano_size
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

        # modify homographies from the stitcher to map points to world coordinates
        self._world_homo_left = self.world_homo.dot(self.homo_left)
        self._world_homo_right = self.world_homo.dot(self.homo_right)

    def get_parameters(self):
        """Return the estimated or loaded parameters of the Surveyor needed for later stitching.

        With this function you could save the Surveyor parameters and load them later for further
        stitching of images and mapping of image coordinates/angels to hive coordinates/angles in
        relation to hive.
        """
        StitchingParams = collections.namedtuple('SurveyorParams', ['homo_left', 'homo_right',
                                                                    'size_left', 'size_right',
                                                                    'cam_id_left', 'cam_id_right',
                                                                    'origin', 'ratio_px_mm',
                                                                    'pano_size'])
        result = StitchingParams(self.homo_left, self.homo_right,
                                 self.size_left, self.size_right,
                                 self.cam_id_l, self.cam_id_r,
                                 self.origin, self.ratio_px_mm,
                                 self.pano_size)
        return result

    def load_parameters(self, homo_left):
        pass

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
        size_left = self.size_left
        size_right = self.size_right
        pano_size = self.pano_size

        stitch = stitcher.Stitcher(self.config)

        # using modified homographies to map points and angles to world coordinates
        stitch.load_parameters(self._world_homo_left, self._world_homo_right,
                               size_left, size_right,
                               pano_size)

        if cam_id == self.cam_id_l:
            stitch.map_left_points_angles(points, angles)
        elif cam_id == self.cam_id_r:
            stitch.map_right_points_angles(points, angles)
        else:
            raise Exception('Got invalid cam_id {invalid_ID}, '
                            'cam_id must be {left_ID} or {right_ID}.'.format(invalid_ID=cam_id,
                                                                             left_ID=self.cam_id_l,
                                                                             right_ID=self.cam_id_r)
                            )
