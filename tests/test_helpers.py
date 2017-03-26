import numpy as np
import numpy.testing as npt

import bb_stitcher.helpers as helpers


def test_align_to_display_area():
    # TODO(gitmirgut): Add test with rotation.
    size_left = (4, 3)
    size_right = (5, 2)
    homo_left = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    homo_right = np.float32([[1, 0, 3], [0, 1, 2], [0, 0, 1]])
    homo_trans, display_size = helpers.align_to_display_area(
        size_left, size_right, homo_left, homo_right)

    target_homo = np.float32(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])
    target_size = (8, 4)

    npt.assert_equal(homo_trans, target_homo)
    assert display_size == target_size
