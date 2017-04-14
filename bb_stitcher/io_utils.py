from abc import ABCMeta, abstractmethod
import csv

import numpy as np


class FileHandler(metaclass=ABCMeta):

    def __init__(self):
        self.surveyor = None

    def visit_surveyor(self, surveyor):
        self.surveyor = surveyor

    @abstractmethod
    def save(self, path):
        """Save data to file."""

    @abstractmethod
    def load(self, path):
        """Load data from file."""


class NPZHandler(FileHandler):

    def save(self, path):
        np.savez(path,
                 homo_left=self.surveyor.homo_left,
                 homo_right=self.surveyor.homo_right,
                 size_left=self.surveyor.size_left,
                 size_right=self.surveyor.size_right,
                 cam_id_left=self.surveyor.cam_id_left,
                 cam_id_right=self.surveyor.cam_id_right,
                 origin=self.surveyor.origin,
                 ratio_px_mm=self.surveyor.ratio_px_mm,
                 pano_size=self.surveyor.pano_size
                 )

    def load(self, path):
        with np.load(path) as data:
            self.surveyor.homo_left = data['homo_left']
            self.surveyor.homo_right = data['homo_right']
            self.surveyor.size_left = tuple(data['size_left'])
            self.surveyor.size_right = tuple(data['size_right'])
            self.surveyor.cam_id_left = data['cam_id_left']
            self.surveyor.cam_id_right = data['cam_id_right']
            self.surveyor.origin = data['origin']
            self.surveyor.ratio_px_mm = data['ratio_px_mm']
            self.surveyor.pano_size = tuple(data['pano_size'])

class CSVHandler(FileHandler):

    def save(self, path):
        with open(path, 'w', newline='') as csvfile:
            writer=csv.DictWriter
