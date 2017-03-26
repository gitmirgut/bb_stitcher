"""Image Stitcher especially designed for the BeesBook Project."""
import cv2
import numpy as np

import bb_stitcher.helpers as helpers


class Stitcher(object):
    """Class to create a 'panorama' from two images."""

    def __init__(self, homo_l=None, homo_r=None):
        """"Initialize the stitcher."""

    def estimate_transformation(self):
        """Update self.homo_l and homo_r to new values.

        This should be overridden by a sublcass to customize stitching.
        Return the transformation matrix for the left and right image.
        """
        raise NotImplemented()

    def compose_panorama(self, left=None, right=None):
        """Try to compose the given images into the final pano.

        This happens under the assumption that the image transformations were estimated or loaded
        before.
        """
        pass

    def stitch(self, left_image, right_image):
        """Try to stitch the given images."""
        pass


class FeatureBasedStitcher(Stitcher):
    """Class to create a feature based stitcher."""

    def __init__(self, config):
        """Initialize a feature based stitcher.

        Args:
            config: config file which holds the basic stitching parameters.
        """
        # TODO(gitmirgut) add autoload config file
        self.overlap = int(config['FeatureBasedStitcher']['OVERLAP'])
        self.border_top = int(config['FeatureBasedStitcher']['BORDER_TOP'])
        self.border_bottom = int(config['FeatureBasedStitcher']['BORDER_BOTTOM'])
        self.transform = config['FeatureBasedStitcher']['TRANSFORM']
        self.hessianThreshold = float(config['SURF']['HESSIANTHRESHOLD'])
        self.nOctaves = int(config['SURF']['N_OCTAVES'])
        self.max_shift_y = int(config['FeatureMatcher']['MAX_SCHIFT_Y'])

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

    def estimate_transformation(self, image_left, image_right):
        """Estimate transformation for stitching of images based on feature matching.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.

        Returns:
            - **homo_left** (ndarray) -- homography *(3,3)* of the left image to form a panorama.
            - **homo_right** (ndarray) -- homography *(3,3)* of the right image to form a panorama.
            - **pano_size** (tuple) -- Size *(width, height)* of the panorama.
        """
        size_left = image_left.shape[:2][:: - 1]
        size_right = image_right.shape[:2][:: - 1]

        # calculates the mask which will mark the feature searching area.
        mask_left, mask_right = self._calc_feature_mask(
            size_left, size_right, self.overlap, self.border_top, self.border_bottom)

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
            return
        homo_right = cv2.invertAffineTransform(homo_right)
        homo_right = np.vstack([homo_right, [0, 0, 1]])

        homo_left = np.float64(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])

        homo_trans, pano_size = helpers.align_to_display_area(
            size_left, size_right, homo_left, homo_right)

        self.homo_left = homo_trans.dot(homo_left)
        self.homo_right = homo_trans.dot(homo_right)
        self.pano_size = pano_size
        return self.homo_left, self.homo_right, self.pano_size
