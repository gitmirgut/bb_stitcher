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
from abc import ABCMeta
from abc import abstractmethod
import ast
import csv
import os

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
            fieldnames = ['homo_left', 'homo_right',
                          'size_left', 'size_right',
                          'cam_id_left', 'cam_id_right',
                          'origin', 'ratio_px_mm',
                          'pano_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerow({
                'homo_left': self.surveyor.homo_left.tolist(),
                'homo_right': self.surveyor.homo_right.tolist(),
                'size_left': self.surveyor.size_left,
                'size_right': self.surveyor.size_right,
                'cam_id_left': self.surveyor.cam_id_left,
                'cam_id_right': self.surveyor.cam_id_right,
                'origin': self.surveyor.origin.tolist(),
                'ratio_px_mm': self.surveyor.ratio_px_mm,
                'pano_size': self.surveyor.pano_size,
            })

    def load(self, path):
        with open(path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                self.surveyor.homo_left = np.array(ast.literal_eval(row['homo_left']))
                self.surveyor.homo_right = np.array(ast.literal_eval(row['homo_right']))
                self.surveyor.size_left = ast.literal_eval(row['size_left'])
                self.surveyor.size_right = ast.literal_eval(row['size_right'])
                self.surveyor.cam_id_left = int(row['cam_id_left'])
                self.surveyor.cam_id_right = int(row['cam_id_right'])
                self.surveyor.origin = np.array(ast.literal_eval(row['origin']))
                self.surveyor.ratio_px_mm = float(row['ratio_px_mm'])
                self.surveyor.pano_size = ast.literal_eval(row['pano_size'])
                break


def get_file_handler(path):
    __, ext = os.path.splitext(path)
    if ext == '.npz':
        filehandler = NPZHandler()
    elif ext == '.csv':
        filehandler = CSVHandler()
    else:
        raise Exception('File format with "{ext}" not supported'.format(ext=ext))
    return filehandler
