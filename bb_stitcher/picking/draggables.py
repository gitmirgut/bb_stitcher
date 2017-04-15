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
"""This module contains draggable objects for the matplotlib GUI.

The objects are used to mark specific points on an image. The module also
contains a special list to save these objects. This list is extended with
special functions to get the coordinates of the marked points.
"""
import cv2
import numpy as np


class DraggableMark(object):
    """Defines marks which can be dragged by mouse.

    The placed mark can be dragged by simple left click and can be **refined** by pressing the
    :const:`key_refine` specific button and they can be marked as **selected** by pressing
    :const:`key_select`.
    """

    _lock = None  # only one mark at at time can be animated.
    key_refine = 'r'  # if this key is pressed the mark will be refined
    key_select = 's'  # if this key is pressed the mark will be selected

    def __init__(self, mark, img=None):
        """Initialize a draggable mark."""
        self.mark = mark
        self.img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        self.press = None
        self.background = None
        self.selected = False

    def get_coordinate(self):
        """Return the current coordinate of the mark.

        Returns:
            ndarray: center of the mark *(2,)*.
        """
        return self.mark.get_xydata()[0]

    def _toggle_select(self):
        """Toggle mark selection state."""
        if self.selected is False:
            self.selected = True
            self.mark.set_color('g')
        else:
            self.selected = False
            self.mark.set_color('r')

    def _select(self):
        self.selected = True
        self.mark.set_color('g')

    def _unselect(self):
        self.selected = False
        self.mark.set_color('r')

    def _refine(self):
        """Refine the location of the mark.

        Use it if you want that the mark should be on a corner.
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        new_coordinate = self.get_coordinate()
        new_coordinate = np.array([[new_coordinate]], dtype=np.float32)
        cv2.cornerSubPix(self.img, new_coordinate, (40, 40), (-1, -1), criteria)
        x, y = new_coordinate[0][0]
        self.mark.set_xdata(x)
        self.mark.set_ydata(y)
        self._unselect()
        self.mark.set_color('y')
        self.mark.figure.canvas.draw()

    def connect(self):
        """Connect to the mark to various Events."""
        self.cid_press = self.mark.figure.canvas.mpl_connect(
            'button_press_event', self._on_press)
        self.cid_release = self.mark.figure.canvas.mpl_connect(
            'button_release_event', self._on_release)
        self.cid_motion = self.mark.figure.canvas.mpl_connect(
            'motion_notify_event', self._on_motion)
        self.cid_key = self.mark.figure.canvas.mpl_connect(
            'key_release_event', self._on_key)

    def _on_press(self, event):
        """Check on button press if mouse is over this DraggableMarks."""
        # if the event is not in the axes return
        if event.inaxes != self.mark.axes:
            return

        # checks if an other DraggableMarks is already chosen
        if DraggableMark._lock is not None:
            return

        # This checks if the mouse is over us (marker)
        contains, __ = self.mark.contains(event)
        if not contains:
            return

        # set back the color to red to mark that is not refined and remove
        # selection flag
        self._unselect()

        # get the current coordinates of the marker
        x, y = self.get_coordinate()

        # cache the coordinates and the event coordinates
        self.press = x, y, event.xdata, event.ydata

        # Locks the dragging of other DraggableMarker
        DraggableMark._lock = self

        # draw everything but the selected marker and store the pixel buffer
        canvas = self.mark.figure.canvas
        axes = self.mark.axes
        self.mark.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.mark.axes.bbox)

        # now redraw just the marker
        axes.draw_artist(self.mark)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def _on_motion(self, event):
        """On motion the mark will move if the mouse is over this marker."""
        if DraggableMark._lock is not self:
            return
        if event.inaxes != self.mark.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.mark.set_xdata(x0 + dx)
        self.mark.set_ydata(y0 + dy)

        canvas = self.mark.figure.canvas
        axes = self.mark.axes

        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.mark)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def _on_release(self, event):
        """On release the press data will be reset."""
        if DraggableMark._lock is not self:
            return

        self.press = None
        DraggableMark._lock = None

        # turn off the mark animation property and reset the background
        self.mark.set_animated(False)
        self.background = None

        # redraw the full figure
        self.mark.figure.canvas.draw()

    def _on_key(self, event):
        """Check what key ist pressed and executes corresponding function."""
        if event.inaxes != self.mark.axes:
            return

        # This checks if the mouse is over us (marker)
        contains, __ = self.mark.contains(event)
        if not contains:
            return

        if event.key == DraggableMark.key_select:
            self._toggle_select()
        elif event.key == DraggableMark.key_refine:
            self._refine()

        self.mark.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all the stored connection ids."""
        self.mark.figure.canvas.mpl_disconnect(self.c_id_press)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_release)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_motion)
        self.mark.figure.canvas.mpl_disconnect(self.cid_key)
        self.mark.figure.canvas.draw()


class DraggableMarkList(list):
    """Extended List with some extra functions for :obj:`DraggableMark` objects."""

    def __init__(self, *args):
        """Initialize a list which holds :obj:`DraggableMark` objects."""
        list.__init__(self, *args)

    def get_points(self, all_pts=True):
        """Convert the list of :obj:`DraggableMark` objects to a ndarray holding just coordinates.

        Args:
            all_pts (bool): if ``True`` function returns all coordinate. If ``False`` function \
            return just coordinates form :obj:`DraggableMark` objects marked as selected.

        Returns:
            ndarray: array that contains just the coordinates of the :obj:`DraggableMark` objects \
            of the list.
        """
        dms = []
        if all_pts:
            dms = self
        else:
            for i, dm in enumerate(self):
                if dm.selected:
                    dms.append(dm)
        points = np.zeros((len(dms), 2), np.float32)
        for i, dm in enumerate(dms):
            points[i] = dm.get_coordinate()

        return points
