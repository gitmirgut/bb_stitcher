# bb_stitcher: Stitcher/Surveyor for the BeesBook images.
[![Build Status](https://travis-ci.org/gitmirgut/bb_stitcher.svg?branch=master)](https://travis-ci.org/gitmirgut/bb_stitcher)
[![Coverage Status](https://coveralls.io/repos/github/gitmirgut/bb_stitcher/badge.svg?branch=master)](https://coveralls.io/github/gitmirgut/bb_stitcher?branch=master)
[![Docker Automated buil](https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg)](https://hub.docker.com/r/gitmirgut/bb_stitcher/)
[![Documentation Status](https://readthedocs.org/projects/bb-stitcher/badge/?version=latest)](http://bb-stitcher.readthedocs.io/en/latest/?badge=latest)

* **Documentation:** http://bb-stitcher.readthedocs.io/
* **Source:** https://github.com/BioroboticsLab/bb_stitcher

## (special) Requirements
The following requirements must be installed manually and cannot be installed by pip:
* [OpenCV3](https://github.com/opencv/opencv)
* [opencv_contrib](https://github.com/opencv/opencv_contrib)

[Good Instruction](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) for installing opencv with opencv_contrib package.
##Installation from source

Install dependencies using:

```bash
$ pip install -r requirements.txt
```
Then, install bb_stitcher using:

```
$ pip install .
```