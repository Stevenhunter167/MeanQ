from __future__ import annotations
import os
import glob
import csv
import json
import bz2
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class ExperimentPath:

    """ Helper class for handling experiment logging and plotting """

    @staticmethod
    def now():
        return datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")

    def __init__(self, path):
        self._path = path

    def __str__(self) -> str:
        return self._path

    def str(self) -> str:
        """ explicit conversion to string """
        return self._path

    # basic file selection

    def __getitem__(self, item: str) -> ExperimentPath:
        """ enter sub-folder """
        os.makedirs(self._path, exist_ok=True)
        return ExperimentPath(os.path.join(self._path, item))

    # checks and verify

    def isfile(self, file_type: str) -> bool:
        """ check if current path is a file """
        return os.path.isfile(self._path + '.' + file_type)

    def isdir(self) -> bool:
        """ check if current path is a directory """
        return os.path.isdir(self._path)

    def ensure_path_existance(self):
        """ ensure the path to current file is valid """
        os.makedirs(os.path.split(self._path)[0], exist_ok=True)

    # traverse folder

    def iglob(self, query: str) -> list:
        """ list all query matches using iglob """
        return list(glob.iglob(self._path + '/' + query))

    def listdir(self) -> list:
        """ list all files or folders in current directory """
        return os.listdir(self._path)

    # saving and loading csv files

    def with_type(self, filetype):
        """ return path with file type postfix """
        return self._path + (f".{filetype}" if not self._path.endswith(f".{filetype}") else '')

    def csv_writerow(self, row: list) -> None:
        self.ensure_path_existance()
        with open(self.with_type('csv'), 'a') as f:
            csv.writer(f).writerow(row)

    def csv_read(self, nrows: int) -> np.ndarray:
        with open(self.with_type('csv'), 'r') as f:
            return pd.read_csv(f, nrows=nrows, header=None).to_numpy()

    def csv_plot(self, nrows: int, x: int, y: int, label: str, color: str):
        with open(self.with_type('csv'), 'r') as f:
            data = pd.read_csv(f, nrows=nrows, header=None).to_numpy()
            plt.plot(data[:, x], data[:, y], label=label, color=color)

    # saving and loading json files

    def json_read(self) -> dict:
        with open(self.with_type('json'), 'r') as f:
            return json.load(f)

    def json_write(self, d: dict) -> None:
        self.ensure_path_existance()
        with open(self.with_type('json'), 'w') as f:
            json.dump(d, f)

    # saving and loading txt files

    def txt_write(self, s: str) -> None:
        self.ensure_path_existance()
        with open(self.with_type('txt'), 'w') as f:
            f.write(s)

    # saving and loading objects

    def save_model(self, obj: object) -> None:
        torch.save(obj, self.str())

    def load_model(self, map_location=None) -> object:
        return torch.load(self.str(), map_location=map_location)

    def save(self, obj: object, compress=False) -> None:
        self.ensure_path_existance()
        if compress:
            with bz2.open(self._path + '.pkl', 'wb') as f:
                pickle.dump(obj, f)
        else:
            with open(self._path + '.pkl', 'wb') as f:
                pickle.dump(obj, f)

    def load(self, compress=False) -> object:
        path = self._path + '.pkl' if not self._path.endswith('.pkl') else self._path
        if compress:
            with bz2.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)

    # aggregate data matrix

    def sync_stack(self, labels, queries, morethan=0):
        """ stack syncronously recorded data """
        data = dict()
        for label, query in zip(labels, queries):
            mints = float("inf")
            maxts = float("-inf")
            maxi  = -1
            dataframes = []
            paths = list(self.iglob(query))
            # print(query)
            # for path in paths:
            #     print(path)
            # print()
            print(query)
            for path in paths:
                dataframe = ExperimentPath(path).csv_read(nrows=None)
                # maxts = max(maxts, dataframe.shape[0])
                if dataframe.shape[0] > morethan:
                    mints = min(mints, dataframe.shape[0])
                    dataframes.append(dataframe)
                    print(path, dataframe.shape[0])
                # else:
                    # print(path, dataframe.shape[0])
            # print()
            for i in range(len(dataframes)):
                dataframes[i] = dataframes[i][:mints, :]
            # for i in range(len(dataframes)):
            #     dataframes[i] = np.concatenate(
            #         [dataframes[i], 
            #         np.empty((maxts - dataframes[i].shape[0], dataframes[i].shape[1]))], 
            #         axis=0)
            #     assert dataframes[i].shape[0] == maxts
            data[label] = np.stack(dataframes)
        return data

    def async_stack(self, labels, queries):
        data = dict()
        for label, query in zip(labels, queries):
            dataframes = []
            for path in self.iglob(query):
                dataframe = ExperimentPath(path).csv_read(nrows=None)
                dataframes.append(dataframe)
            data[label] = dataframes
        return data

    def savefig(self):
        self.ensure_path_existance()
        plt.savefig(self._path)
        plt.close()


if __name__ == '__main__':
    # demo
    exp = ExperimentPath('exp/regular/asterix')
    # exp['abc']['x'].csv_writerow([1, 2, 3])
    # exp['bcd']['x'].csv_writerow([1, 2, 3])
    # print(exp['abc'].listdir())
    # print(list(exp.iglob('*/x.csv')))
    # print(exp.now())
    
    
    # print(data['DQN'].shape)

