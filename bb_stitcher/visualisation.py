#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.
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
                       markerSize=40, thickness=1)


def draw_grid(image, origin, ratio_px_mm, step_size_mm=8):
    """Draw a grid with axes in mm on the image.

    Args:
        image (ndarray): Image to draw on.
        origin (ndarray): The orgin of the grid / axes.
        ratio_px_mm: Ratio to convert pixel to mm.
        step_size_mm: The (step) distance between the grid lines.
    """
    w, h = image.shape[:2][::-1]
    x, y = origin
    step_size_px = step_size_mm / ratio_px_mm

    # draw vertical lines in mm
    max_lines_vert = int((w - x) / step_size_px)
    for i in range(max_lines_vert):
        pt1_v = np.zeros((2,), dtype=np.uint16)
        pt1_v[0] = x + i * step_size_px
        pt1_v[1] = y
        pt1_v = tuple(pt1_v)
        pt2_v = (int(pt1_v[0]), h)
        cv2.line(image, pt1_v, pt2_v, color=(255, 0, 0), thickness=4)
        cv2.putText(image, str(i * step_size_mm), pt1_v, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    thickness=4)

    # draw horizontal lines and distance in mm
    max_lines_hori = int((h - y) / step_size_px)
    for i in range(max_lines_hori):
        pt1_h = np.zeros((2,), dtype=np.uint16)
        pt1_h[0] = x
        pt1_h[1] = y + i * step_size_px
        pt1_h = tuple(np.uint16(pt1_h))
        pt2_h = (w, int(pt1_h[1]))
        cv2.line(image, pt1_h, pt2_h, color=(255, 0, 0), thickness=4)
        cv2.putText(image, str(i * step_size_mm), pt1_h, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                    thickness=4)
