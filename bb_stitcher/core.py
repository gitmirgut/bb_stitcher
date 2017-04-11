"""Module to connect stitcher and mapping from image coordinates to world coordinates."""
import bb_stitcher.stitcher as stitcher


class Surveyor(object):
    """Class to determine the relationship between two images of one comb side .

    The ``Surveyor`` determines all needed data to stitch two images from different areas of one
    comb side to a complete view of the comb. On this basis the ``Surveyor`` can be used to map the
    coordinates from these images to hive coordinates.
    """

    def __init__(self, config, stitcher_type=None):
        """Initialize Surveyor."""
        self.config = config
        self.stitcher_type = stitcher_type

    def determine_mapping_parameters(self, img_l, img_r, angl_l, angl_r,
                                     cam_id_l, cam_id_r, stitcher_type):
        """Determine the parameters to mapping parameters.

        This functions is used to calculate all needed data to stitch two images and to map
        image coordinates and angels to hive coordinates.

        Args:
            img_l:
            img_r:
            angl_l:
            angl_r:
            cam_id_l:
            cam_id_r:
            stitcher_type:
        """
        stitcher.RectangleStitcher(self.config)
