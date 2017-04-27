--------------------
Estimate Parameters
--------------------

For the Estimation of the parameters for stitching/surveying of two images
of one comb side, you could use the ``bb_stitcher`` command from commandline.
The following command gives you an overview of the possible commands::

    $ bb_stitcher -h

If you want to estimate the parameters for the stitching, you have to use::

    $ bb_stitcher estimate

To see all the positional arguments needed, use ``$bb_stitcher estimate -h``::

    positional arguments:
    {fb,rect}    Define the stitcher to use:
                    fb - FeatureBasedStitcher
                    rect - RectangleStitcher
    left         Path of the left image.
    right        Path of the right image.
    left_angle   Rotation angle of the left image (counter-clockwise).
    right_angle  Rotation angle of the right image (counter-clockwise).
    left_camID   Cam ID of the camera which shot the left image.
    right_camID  Cam ID of the camera which shot the right image.
    out          Output path of the stitching data.
                 Supported Types: .npz,.csv

As you can see there are two options for estimation of the parameters. ``fb``-FeatureBasedStitcher
and ``rect``-RectangleSticher. These are two different approaches, the
`FeatureBasedStitcher <https://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Bachelor-theses/2016/struempel/Bachelor-Struempel.pdf#figure.caption.24>`_ is based
on Feature Detection and Feature Matching. The second one so-called
`RectangleStitcher <https://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Bachelor-theses/2016/struempel/Bachelor-Struempel.pdf#figure.caption.27>`_ maps special
marked points on the comb border to an abstracted rectangle.

**Example:** ::

    $bb_stitcher estimate fb Cam_0_2016-09-01T12:56:50.801920Z.jpg Cam_1_2016-09-01T12:56:50.801926Z.jpg 90 -90 0 1 parameters.csv

------------------

Steps of Estimation
^^^^^^^^^^^^^^^^^^^
The process of parameter estimation is divided in 3 steps:

1. Step: Estimate parameters to form a panorama.
2. Step: Determine the origin.
3. Step: Determine the ratio between pixels an mm.