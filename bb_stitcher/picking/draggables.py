import cv2
import numpy as np


class DraggableMarks(object):
    """Defines Marker which can be dragged by mouse.

    The placed mark can be dragged by simple left click and can be refined
    by pressing the specif button.
    """

    lock = None  # only one mark at at time can be animated.

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
        """Connect to all needed Events."""
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
        if DraggableMarks.lock is not None:
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
        DraggableMarks.lock = self

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
        if DraggableMarks.lock is not self:
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
        if DraggableMarks.lock is not self:
            return

        self.press = None
        DraggableMarks.lock = None

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

        if event.key == 's':
            self._toggle_select()
        elif event.key == 'r':
            self._refine()

        self.mark.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all the stored connection ids."""
        self.mark.figure.canvas.mpl_disconnect(self.c_id_press)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_release)
        self.mark.figure.canvas.mpl_disconnect(self.c_id_motion)
        self.mark.figure.canvas.mpl_disconnect(self.cid_key)
        self.mark.figure.canvas.draw()