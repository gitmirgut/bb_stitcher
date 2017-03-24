"""Image Stitcher especially designed for the BeesBook Project."""


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

    def estimate_transformation(self):
        """Estimate transformation of images based on Feature Matching."""
        pass


if __name__ == '__main__':
    pass
