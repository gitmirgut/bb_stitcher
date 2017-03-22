from abc import ABCMeta, abstractmethod


class Stitcher(metaclass=ABCMeta):
    """Class to create a panorama from two images."""

    @abstractmethod
    def estimate_transformation(self):
        """Return the transformation matrix for the left and right image."""


class FeatureBassedStitcher(Stitcher):

    def estimate_transformation(self):
        pass


if __name__ == '__main__':
    fb = FeatureBassedStitcher()
    fb.estimate_transformation()