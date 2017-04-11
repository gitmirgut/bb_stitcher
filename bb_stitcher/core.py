"""Module to connect stitcher and mapping from image coordinates to world coordinates."""


class Surveyor(object):
    """Class to determine the relationship between two images of one comb side .

    The ``Surveyor`` determines all needed data to stitch two images from different areas of one
    comb side to a complete view of the comb. On this basis the ``Surveyor`` can be used to map the
    coordinates from these images to hive coordinates.
    """

    def __init__(self, config):
        """Initialize Surveyor."""
        self.config = config

    def determine_mapping_parameters(self, img_l, img_r, angl_l, angl_r,
                                     cam_id_l, cam_id_r, stitcher_type):
        """Determine the parameters to mapping parameters.

        This functions is used to calculate all needed data to stitch two images and to map
        image coordinates and angels to hive coordinates.

        Args:
            img_l (str): Path to the left image.
            img_r (str): Path to the right image.
            angl_l (int): Angle in degree to rotate left image.
            angl_r (int): Angle in degree to rotate right image.
            cam_id_l (int): ID of the camera, which shot the left image.
            cam_id_r (int): ID of the camera, which shot the right image.
            stitcher_type (Stitcher): Stitcher to use for stitching of the images.
        """
        pass
