import cv2
import numpy as np
import os
import pytest

import bb_stitcher.visualisation as visualisation


@pytest.fixture
def outdir(main_outdir):
    out_path = os.path.join(main_outdir, str(__name__))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


@pytest.fixture
def sample():
    return np.ones((310, 310, 3), dtype=np.uint8) * 255


def test_draw_arrows(sample, outdir):
    positions = np.ones((9, 2)) * 155
    angles = np.ones((9,))
    for i, val in enumerate(range(- 4, 5)):
        angles[i] = val * np.pi / 4
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
        [450, 450],
        [750, 450],
        [150, 750],
        [450, 750],
        [750, 750],
    ])
    angles = np.ones((9,))
    for i, val in enumerate(range(-4, 5)):
        angles[i] = val * np.pi / 4
    visualisation.draw_complex_marks(image, positions, angles)
    out = os.path.join(outdir, 'draw_complex.jpg')
    cv2.imwrite(out, image)


def test_draw_grid(panorama, outdir):
    origin = np.array([94.43029022, 471.89901733], dtype=np.float32)
    ratio = 0.0644410123918
    visualisation.draw_grid(panorama, origin, ratio, step_size_mm=8)
    out = os.path.join(outdir, 'draw_grid.jpg')
    cv2.imwrite(out, panorama)
