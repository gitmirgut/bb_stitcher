"""This module contains various image stitchers especially designed for the BeesBook Project."""
import collections

import cv2
import numpy as np

import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker as picker
import bb_stitcher.prep as prep


class Stitcher(object):
    """Class to create a 'panorama' from two images."""

    def __init__(self, config=None, rectify=True):
        """"Initialize the stitcher."""
        if config is None:
            self.config = helpers.get_default_config()
        else:
            self.config = config
        self.rectify = rectify
        if rectify:
            self.rectificator = prep.Rectificator(self.config)
        self.homo_left = None
        self.homo_right = None
        self.size_left = None
        self.size_right = None
        self.pano_size = None

    def _prepare_image(self, image, angle=0):
        """Prepare image for stitching.

        It rotates and rectifies the image. Ff the Stitcher is initialized with ``rectify=False``
        the image will not be rectified.

        Args:
            image (ndarray): Image to prepare.
            angle (int): angle in degree to rotate image.

        Returns:
            - **image** (ndarray) -- rotated (and rectified) image.
            - **affine** (ndarray) -- An affine *(3,3)*--matrix for rotation of image or points.
        """
        image = helpers.add_alpha_channel(image)
        if self.rectify:
            image = self.rectificator.rectify_image(image)
        image_rot, affine = prep.rotate_image(image, angle)
        return image_rot, affine

    def load_parameters(self, homo_left=None, homo_right=None, size_left=None, size_right=None,
                        pano_size=None):
        """Load needed parameters for stitching points, angles and images.

         This function becomes handy if you calculate the parameters in an earlier stitching
         process and did not want to calculate the parameters again and just want to map points,
         angles or images which were made under the same camera setup as the earlier stitching
         process.

        Args:
            homo_left (ndarray): homography *(3,3)* for data from the left side to form a panorama.
            homo_right (ndarray): homography *(3,3)* for data from the right side to form a \
            panorama.
            size_left (tuple): Size of the left image, which was used to calculate homography.
            size_right (tuple): Size of the right image, which was used to calculate homography.
            pano_size (tuple): Size of the panorama.
        """
        self.homo_left = homo_left
        self.homo_right = homo_right
        self.size_left = size_left
        self.size_right = size_right
        self.pano_size = pano_size

    def get_parameters(self):
        """Return the estimated or loaded parameters of the stitcher needed for later stitching.

        With this function you could save the stitching parameters and load them later for further
        stitching of points and angles (see ``set_parameters``).

        Use this function if you estimated the transform and did not want to estimate the parameters
        """
        StitchingParams = collections.namedtuple('StichingParams', ['homo_left', 'homo_right',
                                                                    'size_left', 'size_right',
                                                                    'pano_size'])
        result = StitchingParams(self.homo_left, self.homo_right,
                                 self.size_left, self.size_right,
                                 self.pano_size)
        return result

    def estimate_transform(self, image_left, image_right, angle_left=0, angle_right=0):
        """Estimate transformation/homography of the left and right images/data to form a panorama.

        Return the transformation matrix for the left and right image.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.
            angle_left (int): Angle in degree to rotate left image.
            angle_right (int): Angle in degree to rotate right image.

        Warning:
            This must be overridden by a sublcass to customize stitching.
        """
        raise NotImplementedError()

    def compose_panorama(self, image_left, image_right):
        """Try to compose the given images into the final panorama.

        This happens under the assumption that the image transformations were estimated or loaded
        before.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.

        Returns:
            ndarray: panorama (stitched image)
        """
        image_left = helpers.add_alpha_channel(image_left)
        image_right = helpers.add_alpha_channel(image_right)

        if self.rectify:
            image_left = self.rectificator.rectify_image(image_left)
            image_right = self.rectificator.rectify_image(image_right)

        image_left = cv2.warpPerspective(image_left, self.homo_left, self.pano_size)
        image_right = cv2.warpPerspective(image_right, self.homo_right, self.pano_size)

        alpha = 0.5
        cv2.addWeighted(image_left, alpha, image_right, 1 - alpha, 0, image_left)

        return image_left

    def map_left_points(self, points):
        """Map points from the left image to the panorama.

        This happens under the assumption that the image transformations were estimated or loaded
        before.

        Args:
            points (ndarray(float)): List of points from left image *(N,2)*.

        Returns:
            ndarray: ``points`` mapped to panorama *(N,2)*
        """
        # TODO(gitmirgut): PoC auto convert to float
        # TODO(gitmirgut): Add exception if size is None
        if self.rectify:
            points = self.rectificator.rectify_points(points, self.size_left)
        points = np.array([points])
        return cv2.perspectiveTransform(points, self.homo_left)[0]

    def map_left_points_angles(self, points, angles):
        """Map points and angles from the left image to the panorama.

        This happens under the assumption that the image transformations were estimated or loaded
        before.

        Args:
            points (ndarray(float)): List of points from left image *(N,2)*.
            angles (ndarray): Angles in rad (length *(N,)*).

        Returns:
            - **points_mapped** (ndarray) -- ``points`` mapped to panorama *(N,2)*
            - **angles_mapped** (ndarray) -- ``angles`` mapped to panorama *(N,)*
        """
        angle_pt_repr = helpers.angles_to_points(points, angles)
        if self.rectify:
            points = self.rectificator.rectify_points(points, self.size_left)
            angle_pt_repr = self.rectificator.rectify_points(angle_pt_repr, self.size_left)
        points = np.array([points])
        angle_pt_repr = np.array([angle_pt_repr])
        points_mapped = cv2.perspectiveTransform(points, self.homo_left)[0]
        angle_pt_repr_mapped = cv2.perspectiveTransform(angle_pt_repr, self.homo_left)[0]
        angles_mapped = helpers.points_to_angles(points_mapped, angle_pt_repr_mapped)
        return points_mapped, angles_mapped

    def map_right_points(self, points):
        """Map points from the right image to the panorama.

        This happens under the assumption that the image transformations were estimated or loaded
        before.

        Args:
            points (ndarray(float)): List of points from right image *(N,2)*.

        Returns:
            ndarray: ``points`` mapped to panorama *(N,2)*
        """
        # TODO(gitmirgut): PoC auto convert to float
        if self.rectify:
            points = self.rectificator.rectify_points(points, self.size_right)
        points = np.array([points])
        return cv2.perspectiveTransform(points, self.homo_right)[0]

    def map_right_points_angles(self, points, angles):
        """Map points and angles from the right image to the panorama.

        This happens under the assumption that the image transformations were estimated or loaded
        before.

        Args:
            points (ndarray(float)): List of points from right image *(N,2)*.
            angles (ndarray): Angles in rad (length *(N,)*).

        Returns:
            - **points_mapped** (ndarray) -- ``points`` mapped to panorama *(N,2)*
            - **angles_mapped** (ndarray) -- ``angles`` mapped to panorama *(N,)*
        """
        angle_pt_repr = helpers.angles_to_points(points, angles)
        if self.rectify:
            points = self.rectificator.rectify_points(points, self.size_right)
            angle_pt_repr = self.rectificator.rectify_points(angle_pt_repr, self.size_right)
        points = np.array([points])
        angle_pt_repr = np.array([angle_pt_repr])
        points_mapped = cv2.perspectiveTransform(points, self.homo_right)[0]
        angle_pt_repr_mapped = cv2.perspectiveTransform(angle_pt_repr, self.homo_right)[0]
        angles_mapped = helpers.points_to_angles(points_mapped, angle_pt_repr_mapped)
        return points_mapped, angles_mapped

    @staticmethod
    def _calc_image_to_world_mat(panorama):
        """Determine the matrix to convert image coordinates to world coordinates.

        The user must select two points on the image. The first point will be the origin and the
        distance between the first and the second point, will be used to determine the ratio
        between px and mm.

        Returns:
             ndarray: homography *(3,3)* to transform image points to world points.
        """
        pt_picker = picker.PointPicker()
        points = pt_picker.pick([panorama], False)
        start_point, end_point = points[0]
        distance_mm = float(input('Distance in mm of the two selected points: '))
        ratio = helpers.get_ratio_px_to_mm(start_point, end_point, distance_mm)

        # define matrix to convert image coordinates to world coordinates
        homo_to_world = np.array([
            [ratio, 0, start_point[0]],
            [0, ratio, end_point[1]],
            [0, 0, 1]], dtype=np.float32)

        return homo_to_world


class FeatureBasedStitcher(Stitcher):
    """Class to create a feature based stitcher."""

    def __init__(self, config=None, rectify=True):
        """Initialize a feature based stitcher.

        Args:
            config: config file which holds the basic stitching parameters.
        """
        super().__init__(config, rectify)
        self.overlap = int(self.config['FeatureBasedStitcher']['OVERLAP'])
        self.border_top = int(self.config['FeatureBasedStitcher']['BORDER_TOP'])
        self.border_bottom = int(self.config['FeatureBasedStitcher']['BORDER_BOTTOM'])
        self.transform = self.config['FeatureBasedStitcher']['TRANSFORM']
        self.hessianThreshold = float(self.config['SURF']['HESSIANTHRESHOLD'])
        self.nOctaves = int(self.config['SURF']['N_OCTAVES'])
        self.max_shift_y = int(self.config['FeatureMatcher']['MAX_SCHIFT_Y'])

    @staticmethod
    def _calc_feature_mask(size_left, size_right, overlap, border_top, border_bottom):
        """Calculate the masks, which defines the area for feature detection.

        The mask is used to shrink the area for searching features.

        Args:
            size_left (tuple): Size of the left image.
            size_right (tuple): Size of the right image.
            overlap (int): Estimated overlap of both images in px.
            border_top (int): Estimated border size on top of both images in px.
            border_bottom (int): Estimated border size on top of both images in px.

        Returns:
            - **mask_left** (ndarray) -- mask area of the left image to search for features.
            - **mask_right** (ndarray) -- mask area of the right image to search for features.
        """
        # TODO(gitmirgut): Add note, why to use border.
        mask_left = np.zeros(size_left[:: - 1], np.uint8)
        mask_right = np.zeros(size_right[:: - 1], np.uint8)
        mask_left[border_top:size_left[1] - border_bottom, size_left[0] - overlap:] = 255
        mask_right[border_top:size_left[1] - border_bottom, :overlap] = 255

        return mask_left, mask_right

    def estimate_transform(self, image_left, image_right, angle_left=0, angle_right=0):
        """Estimate transformation for stitching of images based on feature matching.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.
            angle_left (int): Angle in degree to rotate left image.
            angle_right (int): Angle in degree to rotate right image.
        """
        self.size_left = image_left.shape[:2][::-1]
        self.size_right = image_right.shape[:2][::-1]

        # rectify and rotate images
        image_left, affine_left = self._prepare_image(image_left, angle_left)
        image_right, affine_right = self._prepare_image(image_right, angle_right)

        rot_size_left = image_left.shape[:2][:: - 1]
        rot_size_right = image_right.shape[:2][:: - 1]

        # calculates the mask which will mark the feature searching area.
        mask_left, mask_right = self._calc_feature_mask(
            rot_size_left, rot_size_right, self.overlap, self.border_top, self.border_bottom)

        # Initialize the feature detector and descriptor SURF
        # http://www.vision.ee.ethz.ch/~surf/download.html
        # is noncommercial licensed
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=self.hessianThreshold, nOctaves=self.nOctaves)
        surf.setUpright(True)
        surf.setExtended(128)

        kps_left, ds_left = surf.detectAndCompute(image_left, mask_left)
        kps_right, ds_right = surf.detectAndCompute(image_right, mask_right)

        assert (len(kps_left) > 0 and len(kps_right) > 0)

        # Start with Feature Matching
        bf = cv2.BFMatcher()

        # search the 2 best matches for each left descriptor (ds_left)
        raw_matches = bf.knnMatch(ds_left, ds_right, k=2)

        good_matches = []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance:
                good_match = m[0]
                keypoint_left = np.array(kps_left[good_match.queryIdx].pt)
                keypoint_right = np.array(kps_right[good_match.trainIdx].pt)
                dist = abs(keypoint_left - keypoint_right)

                # checks if the distance in the y direction is to big
                if dist[1] < self.max_shift_y:
                    good_matches.append(good_match)

        good_pts_left = np.float32(
            [kps_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        good_pts_right = np.float32(
            [kps_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        assert len(good_matches) > 2
        homo_right = cv2.estimateRigidTransform(good_pts_left, good_pts_right, False)
        if homo_right is None:
            return None
        homo_right = cv2.invertAffineTransform(homo_right)
        homo_right = np.vstack([homo_right, [0, 0, 1]])

        # include the previous rotation
        homo_left = affine_left
        homo_right = homo_right.dot(affine_right)

        homo_trans, pano_size = helpers.align_to_display_area(
            rot_size_left, rot_size_right, homo_left, homo_right)

        self.homo_left = homo_trans.dot(homo_left)
        self.homo_right = homo_trans.dot(homo_right)
        self.pano_size = pano_size


class RectangleStitcher(Stitcher):
    """Class to create a rectangle stitcher.

    The ``RectangleStitcher`` maps selected points to an abstracted rectangle.
    """

    def estimate_transform(self, image_left, image_right, angle_left=0, angle_right=0):
        """Estimate transformation for stitching of images based on 'rectangle' Stitching.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.
            angle_left (int): Angle in degree to rotate left image.
            angle_right (int): Angle in degree to rotate right image.
        """
        # TODO(gitmirgut) set all to False
        self.size_left = image_left.shape[:2][::-1]
        self.size_right = image_right.shape[:2][::-1]

        image_left, affine_left = self._prepare_image(image_left, angle_left)
        image_right, affine_right = self._prepare_image(image_right, angle_right)

        rot_size_left = image_left.shape[:2][::-1]
        rot_size_right = image_right.shape[:2][::-1]
        pt_picker = picker.PointPicker()
        pts_left, pts_right = pt_picker.pick([image_left, image_right], False)
        assert len(pts_left) == 4 and len(pts_right) == 4
        pts_left_srt = helpers.sort_pts(pts_left)
        pts_right_srt = helpers.sort_pts(pts_right)

        target_pts_left = helpers.raw_estimate_rect(pts_left_srt)
        target_pts_right = helpers.raw_estimate_rect(pts_right_srt)
        target_pts_left, target_pts_right = helpers.harmonize_rects(
            target_pts_left, target_pts_right)

        # declare the shift of the right points
        shift_right = np.amax(target_pts_left[:, 0])
        target_pts_right[:, 0] = target_pts_right[:, 0] + shift_right
        homo_left, __ = cv2.findHomography(pts_left_srt, target_pts_left)
        homo_right, __ = cv2.findHomography(pts_right_srt, target_pts_right)

        # calculate the overall homography including the previous rotation
        homo_left = homo_left.dot(affine_left)
        homo_right = homo_right.dot(affine_right)

        homo_trans, pano_size = helpers.align_to_display_area(
            rot_size_left, rot_size_right, homo_left, homo_right)

        self.homo_left = homo_trans.dot(homo_left)
        self.homo_right = homo_trans.dot(homo_right)
        self.pano_size = pano_size
