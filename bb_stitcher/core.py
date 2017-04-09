import cv2
import numpy as np

import bb_stitcher.helpers as helpers
import bb_stitcher.picking.picker as picker
from bb_stitcher.stitcher import Stitcher


class FeatureBasedStitcher(Stitcher):
    """Class to create a feature based stitcher."""

    def __init__(self, config=None, rectify=True):
        """Initialize a feature based stitcher.

        Args:
            config: config file which holds the basic stitching parameters.
        """
        # TODO(gitmirgut) initialize super()
        if config is None:
            self.config = helpers.get_default_config()
        else:
            self.config = config
        super().__init__(self.config, rectify)
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

        Returns:
            - **homo_left** (ndarray) -- homography *(3,3)* for ``image_left`` to form a panorama.
            - **homo_right** (ndarray) -- homography *(3,3)* for ``image_right`` to form a panorama.
            - **pano_size** (tuple) -- Size *(width, height)* of the panorama.
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
        # TODO(gitmirgut) add note to doc and requirements on license
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

        Returns:
            - **homo_left** (ndarray) -- homography *(3,3)* for ``image_left`` to form a panorama.
            - **homo_right** (ndarray) -- homography *(3,3)* for ``image_right`` to form a panorama.
            - **pano_size** (tuple) -- Size *(width, height)* of the panorama.
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

        # set origin/start and end point for convert measures from px to mm
        self.origin = self.map_left_points(np.array([pts_left_srt[0]]))
        self.end_point = self.map_right_points(np.array([pts_right_srt[1]]))
