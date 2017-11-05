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
"""This module provides different file handlers
to load and save the needed data for the :obj:`.core.Surveyor`."""
from abc import ABCMeta
from abc import abstractmethod
import ast
import csv
import json
import os

import numpy as np

# TODO(gitmirgut): make it more dynamic
valid_ext = ['.npz', '.csv', '.json']


class FileHandler(metaclass=ABCMeta):
    """Abstract base class of FileHandler."""

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
        """Save :obj:`.core.Surveyor` data to numpy file '.npz'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
        np.savez(path,
                 homo_left=self.surveyor.homo_left,
                 homo_right=self.surveyor.homo_right,
                 size_left=self.surveyor.size_left,
                 size_right=self.surveyor.size_right,
                 cam_id_left=self.surveyor.cam_id_left,
                 cam_id_right=self.surveyor.cam_id_right,
                 origin=self.surveyor.origin,
                 ratio_px_mm=self.surveyor.ratio_px_mm
                 )

    def load(self, path):
        """Load :obj:`.core.Surveyor` data to numpy file '.npz'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
        with np.load(path) as data:
            self.surveyor.homo_left = data['homo_left']
            self.surveyor.homo_right = data['homo_right']
            self.surveyor.size_left = tuple(data['size_left'])
            self.surveyor.size_right = tuple(data['size_right'])
            self.surveyor.cam_id_left = data['cam_id_left']
            self.surveyor.cam_id_right = data['cam_id_right']
            self.surveyor.origin = data['origin']
            self.surveyor.ratio_px_mm = data['ratio_px_mm']


class CSVHandler(FileHandler):

    def save(self, path):
        """Save :obj:`.core.Surveyor` data to csv file '.csv'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['homo_left', 'homo_right',
                          'size_left', 'size_right',
                          'cam_id_left', 'cam_id_right',
                          'origin', 'ratio_px_mm']
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
            })

    def load(self, path):
        """Load :obj:`.core.Surveyor` data to csv file '.csv'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
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
                break


class JSONHandler(FileHandler):

    def save(self, path):
        """Save :obj:`.core.Surveyor` data to json file '.json'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
        with open(path, 'w', newline='') as json_file:
            json.dump({
                'homo_left': self.surveyor.homo_left.tolist(),
                'homo_right': self.surveyor.homo_right.tolist(),
                'size_left': self.surveyor.size_left,
                'size_right': self.surveyor.size_right,
                'cam_id_left': self.surveyor.cam_id_left,
                'cam_id_right': self.surveyor.cam_id_right,
                'origin': self.surveyor.origin.tolist(),
                'ratio_px_mm': self.surveyor.ratio_px_mm,
            }, json_file, sort_keys=True, indent=2)

    def load(self, path):
        """Load :obj:`.core.Surveyor` data to json file '.json'.

        Args:
            path (str): Path of the file, which holds the needed data.
        """
        with open(path, 'r', newline='') as json_file:
            json_data = json.load(json_file)
            self.surveyor.homo_left = np.array(json_data['homo_left'])
            self.surveyor.homo_right = np.array(json_data['homo_right'])
            self.surveyor.size_left = json_data['size_left']
            self.surveyor.size_right = json_data['size_right']
            self.surveyor.cam_id_left = int(json_data['cam_id_left'])
            self.surveyor.cam_id_right = int(json_data['cam_id_right'])
            self.surveyor.origin = np.array(json_data['origin'])
            self.surveyor.ratio_px_mm = float(json_data['ratio_px_mm'])


def get_file_handler(path):
    """Returns FileHandler in dependency of file path extension.

    Args:
        path (str): Path of the file.

    Returns:
        :obj:`FileHandler`: file handler for load and save data from surveyor.
    """
    __, ext = os.path.splitext(path)
    if ext == '.npz':
        filehandler = NPZHandler()
    elif ext == '.csv':
        filehandler = CSVHandler()
    elif ext == '.json':
        filehandler = JSONHandler()
    else:
        raise Exception('File format with "{ext}" not supported'.format(ext=ext))
    return filehandler
