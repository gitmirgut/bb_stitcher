"""Image Stitcher especially designed for the BeesBook Project."""
import cv2
import numpy as np


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

    def _get_keypoints_and_descriptors(self, img, mask):
        """Search for Features in <img>."""

        # TODO(gitmirgut) opencv versions check

        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4)
        surf.setUpright(True)
        surf.setExtended(128)

        kps, ds = surf.detectAndCompute(img, mask)

        # if drawMatches:
        #     marked_matches = cv2.drawKeypoints(img, kps, None, (0, 0, 255), 4)
        #     return kps, ds, marked_matches

        return kps, ds

    def estimate_transformation(self, image_left, image_right):
        """Estimate transformation of images based on feature matching.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.
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
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4)
        surf.setUpright(True)
        surf.setExtended(128)

        kps_left, kps_ds = surf.detectAndCompute(image_left, mask_left)
        kps_right, kps_ds = surf.detectAndCompute(image_right, mask_right)

        assert (len(kps_left) > 0 and len(kps_right) > 0)


if __name__ == '__main__':
    pass
