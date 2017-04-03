import cv2
import numpy as np
import os
import pytest

import bb_stitcher.visualisation as visualisation


@pytest.fixture
def sample():
    return np.ones((310, 310, 3), dtype=np.uint8) * 255


def test_draw_arrows(sample, outdir):
    positions = np.ones((8, 2)) * 155
    angles = np.ones((8,))
    print(angles.shape)
    for i in range(- 4, 5):
        angles[i] = i * np.pi / 4
    visualisation.draw_arrows(sample, positions, angles)
    out = os.path.join(outdir, 'draw_arrows.jpg')
    cv2.imwrite(out, sample)


def test_draw_circles(sample, outdir):
    positions = np.array([[75, 155], [150, 155], [225, 155]])
    visualisation.draw_circles(sample, positions)
    out = os.path.join(outdir, 'draw_circles.jpg')
    cv2.imwrite(out, sample)


def test_draw_marks(sample, outdir):
    positions = np.array([[75, 155], [150, 155], [225, 155]])
    visualisation.draw_marks(sample, positions)
    out = os.path.join(outdir, 'draw_marks.jpg')
    cv2.imwrite(out, sample)


def test_draw_complex_marks(outdir):
    image = np.ones((900, 900, 3), dtype=np.uint8) * 255
    positions = np.array([
        [150, 150],
        [450, 150],
        [750, 150],
        [150, 450],
        [750, 450],
        [150, 750],
        [450, 750],
        [750, 750],
    ])
    angles = np.ones((8,))
    for i in range(-4, 5):
        angles[i] = i * np.pi / 4
    visualisation.draw_complex_marks(image, positions, angles)
    out = os.path.join(outdir, 'draw_complex.jpg')
    cv2.imwrite(out, image)
