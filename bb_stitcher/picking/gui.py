"""Initialise GUI to pick various point on an image.

This Module provides a class to initialise a GUI, to pick various points
on one ore multiple images.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import bb_stitcher.helpers as helpers
import bb_stitcher.picking.draggables as draggs


class PointPicker(object):
    """GUI for picking points."""

    def __init__(self, selection=True):
        """Initialise GUI to pick various point on an image."""
        mpl.rcParams['keymap.quit'] = ['ctrl+w', 'cmd+w', 'q']
        mpl.rcParams['keymap.home'] = ['h', 'home']
        mpl.rcParams['keymap.save'] = ['ctrl+s']
        mpl.rcParams['keymap.zoom'] = ['o', 'z']
        self.selection = selection

    def pick(self, images, select=True):
        """Initialise GUI to pick points on multiple images.

        A matplot GUI will be initialised, where the user has to pick 4 points
        on the left and right image. Afterwards the PointPicker will return 2
        clockwise sorted list of the picked points.

        Returns:
            list: Returns a List of len(*image), where each cell contains an ndarray (N,2), which
            holds the coordinates of the selected points per image.
        """
        imgs_a = []
        for img in images:
            imgs_a.append(helpers.add_alpha_channel((img)))
        count_images = len(imgs_a)
        # creating one list per image, which will hold the draggable markers
        # e.g. for 2 images:
        # dms_per_image = [[<dragableMarks first image>],[<dragableMarks second image>]]

        dms_per_image = []
        for __ in range(count_images):
            dms_per_image.append(draggs.DraggableMarkList())

        def _on_click(event):
            # double click left mouse button
            if event.button == 1 and event.dblclick:
                for i, ax in enumerate(axes):
                    if event.inaxes == ax:
                        marker, = ax.plot(
                            event.xdata, event.ydata, 'xr', markersize=10, markeredgewidth=2)
                        dm = draggs.DraggableMark(marker, imgs_a[i])
                        dm.connect()
                        dms_per_image[i].append(dm)
                        fig.canvas.draw()

        fig, axes = plt.subplots(
            nrows=1, ncols=count_images, tight_layout=False)

        # if the nrows == 1 and ncols == 1 the function of plt.subplots returns a single
        # class 'matplotlib.axes._subplots.AxesSubplot' but we want always an array
        if count_images == 1:
            axes = np.array([axes])

        for i, image in enumerate(imgs_a):
            # don't draw y-axis on every image, just on first image
            if i > 0:
                plt.setp(axes[i].get_yticklabels(), visible=False)
            axes[i].imshow(image)

        fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
        points = []
        for i, dms in enumerate(dms_per_image):
            points_per_image = dms.get_points(all=False)
            points.append(points_per_image)
        return points
