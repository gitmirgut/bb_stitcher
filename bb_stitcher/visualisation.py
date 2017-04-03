"""This module contains different functions to draw coordinates and orientations on images."""
import cv2
import numpy as np


# adapted from:
# https://github.com/BioroboticsLab/bb_pipeline/blob/master/pipeline/stages/visualization.py
def draw_arrows(img, positions, angles, color=(0, 0, 255), line_width=6, arrow_length=150):
    u"""Draw arrows from positions in angle direction (clockwise).

    (The 0°-Angle is the x-Axis.)

    Args:
        img (ndarray): Image (min. 3 channel) to draw on.
        positions (ndarray): The points the arrows starts from. *(N,2)*
        angles (ndarray):  Angles in rad (length *(N,)*).
    """
    assert len(positions) == len(angles)
    for i, position in enumerate(positions):
        position = position.astype(np.int32)
        x_to = np.round(position[0] + arrow_length * np.cos(angles[i])).astype(np.int32)
        y_to = np.round(position[1] + arrow_length * np.sin(angles[i])).astype(np.int32)
        cv2.arrowedLine(img, tuple(position), (x_to, y_to), color, line_width, cv2.LINE_AA)


def draw_circles(img, centres, radius=32, color=(0, 0, 255), line_width=6):
    """Draw circles around positions.

    Args:
        img (ndarray): Image (min. 3 channel) to draw on.
        centres (ndarray): The centres of the circles. *(N,2)*
        radius: Radius of the circles.
    """
    for center in centres:
        center = center.astype(np.int32)
        cv2.circle(img, tuple(center), radius, color, line_width)


def draw_marks(img, positions, color=(0, 0, 255), marker_types=cv2.MARKER_CROSS):
    """Draw cross marks on position.

    Args:
        img (ndarray): Image (min. 3 channel) to draw on.
        positions (ndarray): The points to mark. *(N,2)*
    """
    for position in positions:
        cv2.drawMarker(img, tuple(position), color, markerType=marker_types,
                       markerSize=40, thickness=5)


def draw_complex_marks(img, centres, angles, color=(0, 0, 255), marker_types=cv2.MARKER_CROSS):
    """Draw a more complex marks, with circles, marked centres and arrows for angles/drection.

    Args:
        img (ndarray): Image (min. 3 channel) to draw on.
        centres (ndarray): The centres of the marks and starting points of arrows. *(N,2)*
        angles (ndarray):  Angles in rad (length *(N,)*).
    """
    arrow_length = 150
    assert len(centres) == len(angles)
    for i, center in enumerate(centres):
        center = center.astype(np.int32)
        x_to = np.round(center[0] + arrow_length * np.cos(angles[i])).astype(np.int32)
        y_to = np.round(center[1] + arrow_length * np.sin(angles[i])).astype(np.int32)
        cv2.arrowedLine(img, tuple(center), (x_to, y_to), color, thickness=6, line_type=cv2.LINE_AA)
        cv2.circle(img, tuple(center), radius=32, color=color, thickness=6)
        cv2.drawMarker(img, tuple(center), color, markerType=marker_types,
                       markerSize=40, thickness=5)
