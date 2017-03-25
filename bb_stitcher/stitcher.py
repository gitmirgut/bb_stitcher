"""Image Stitcher especially designed for the BeesBook Project."""
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

    @staticmethod
    def _calc_feature_masks_old(left_shape, right_shape, overlap, border):
        """Calculate the mask, which define area for feature detection."""
        left_mask = np.ones(left_shape, np.uint8) * 255
        right_mask = np.ones(right_shape, np.uint8) * 255
        if overlap is not None:
            left_mask[:, :left_shape[1] - overlap] = 0
            right_mask[:, overlap:] = 0
        if border is not None:
            left_mask[:border, :] = 0
            left_mask[left_shape[0] - border:, :] = 0
            right_mask[:border, :] = 0
            right_mask[right_shape[0] - border:, :] = 0

        return left_mask, right_mask

    @staticmethod
    def _calc_feature_mask(size_left, size_right, overlap, border_top, border_bottom):
        """Calculate the masks, which defines the area for feature detection. The mask is used to
        shrink the area for searching features.

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
        mask_left = np.zeros(size_left[::-1], np.uint8)
        mask_right = np.zeros(size_right[::-1], np.uint8)
        mask_left[border_top:size_left[1]-border_bottom, size_left[0]-overlap:] = 255
        mask_right[border_top:size_left[1]-border_bottom, :overlap] = 255

        return mask_left, mask_right

    def estimate_transformation(self, image_left, image_right, config):
        """Estimate transformation of images based on feature matching.

        Args:
            image_left (ndarray): Input left image.
            image_right (ndarray): Input right image.
        """
        pass


if __name__ == '__main__':
    left_mask, right_mask = FeatureBasedStitcher._calc_feature_masks_old((6,8), (6,8), 2, 1)
    mask_left, mask_right = FeatureBasedStitcher._calc_feature_mask((3, 8), (3, 8), 2, 3, 1)
    print(left_mask)
    print(mask_left)

    pass
