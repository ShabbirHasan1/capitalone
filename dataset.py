""" Dataset explorer

"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
import itertools

import utils
import pyutils
import router


def as_dataframe(path):
    """
    Entry point to read data
    Read data and return it as dataframe.
    This will be used across project as to switch reading from any format (e.g. json, csv)
    :param path: full path of the data
    :return:
    """
    df = pd.read_csv(path)
    return df


def dataframe_json_to_csv(path_from: str, path_to: str):
    """
    Converting json dataset file to csv file
    :param path_from:
    :param path_to:
    :return:
    """
    df: pd.DataFrame = pd.read_json(path_from, lines=True)
    df.to_csv(path_to, index=False)


def gen_small_dataset(path: str, nsamples=10000):
    """
    Generate a small dataset from the large one
    :param path: path of the small dataset
    :param nsamples: number of samples to take from large dataset
    :return:
    """
    df: pd.DataFrame = pd.read_csv(router.dataset_large)
    small: pd.DataFrame = df.sample(min(nsamples, len(df)))  # taking min in case nsample is larger than dataset size

    small.to_csv(path, index=False)


class Dataset:
    """ Dataset class data load, store, analyze, explore, visualize etc

    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def _get_data(self) -> pd.DataFrame:
        """
        Get dataframe data that will be used throughout different methods
        :return:
        """
        raise NotImplementedError

    def _process_data(self):
        """
        Read data from outter source (e.g. json, csv file), process it and store in the class
        :return:
        """
        raise NotImplementedError

    def describe(self):
        """
        Describe different features of the dataset
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def as_category(df: pd.DataFrame) -> pd.DataFrame:
        """
        convert all non value column into category
        :param df:
        :return:
        """
        for col in list(df):
            if df[col].dtype not in [np.float, np.int]:
                new_col_name = f'{col}_ascat_dupli'
                df[new_col_name] = df[col].astype('category').cat.codes
        return df

    @staticmethod
    def get_numeric_col(df):
        """
        Get all columns name that has numerical value
        :param df:
        :return:
        """
        cols = []
        for col in list(df):
            if df[col].dtype != bool and pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols

    def _get_path(self, name: str, typ='fig'):
        """
        Get dataset specific path
        : param name: file name
        : param typ: [fig] different type of path
        :return:
        """
        if typ == 'fig':
            path = os.path.join(router.fig, f'{self.name}_{name}')
        else:
            raise ValueError('invalid option')
        return path

    def _structure(self, df=None):
        """
        Basic structure of the data, like number of rows, columns, data types, some sample data etc.
        :param df: passed dataframe. if not passed it'll load default data
        :return:
        """
        if df is None:
            df = self._get_data()

        pyutils.header(f'Basic dataset structure ({self.name})')
        print(f'Number of rows: {len(df)} x_cols: {len(list(df))}')
        print(f'Columns name: {list(df)}')
        pyutils.header(line=f'Basic information:', typ=2)
        print(df.info(null_counts=True))
        pyutils.header(line=f'Sample data:', typ=2)
        print(df.head())

    def _plot_columns(self):
        """
        Plot data of different columns
        :return:
        """
        df = self._get_data()
        for col_name in tqdm(self.get_numeric_col(df), desc='Plotting columns'):
            col = df[col_name]

            # plot histogram
            col.plot.hist(title=col_name)
            path_dist = self._get_path(name=f'{col_name}_dist', typ='fig')
            pyutils.plt_save(path=path_dist)

            # plot scatter plot
            col.reset_index(name=col_name).plot.scatter(x='index', y=col_name, title=col_name)
            path_scatter = self._get_path(name=f'{col_name}_scatter', typ='fig')
            pyutils.plt_save(path=path_scatter)

    def _plot_corr(self, cols: List[str] = (), y_col=None):
        """
        Plot correlation between different columns.
        :param cols: coumns name to use. Ff empty use all column in the dataframe
        :param y_col: print value of y_col column. Print nothing if none is passed
        :return:
        """
        print(f'Plotting correlation between columns..', file=sys.stderr)
        # correlation matrix
        df = self._get_data()
        if len(cols) == 0:
            cols = list(df)

        # find correlation
        corrmat = df[cols].corr()

        # dropping any column if nan is present
        # corrmat = corrmat.dropna()

        # plot correlation matrix
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)

        path = self._get_path(name=f'heatmap', typ='fig')
        pyutils.plt_save(path=path)

        # print y_col corelation with others
        if y_col:
            pyutils.header(f'Correlation of ({y_col}) with other columns', typ=2)
            vals = corrmat[y_col].sort_values()
            print(vals)

    def _plot_pairplot_sns(self, cols: List[str] = (), n_samples=100):
        """
        Pair column plot with sns. Too slow
        :param cols:
        :return:
        """
        print(f'Plotting pairplot columns..', file=sys.stderr)

        df = self._get_data()
        if len(cols) == 0:
            cols = list(df)

        df = df[cols]
        df = df.sample(min(n_samples, len(df)))

        sns.set()
        sns.pairplot(df[cols], height=2.5, diag_kind='kde')

        path = self._get_path(name=f'scatter_matrix', typ='fig')
        pyutils.plt_save(path=path)

    def _plot_pairplot(self, x_cols=[], y_cols=[], n_samples=-1):
        """
        Plot column pairwise
        :param x_cols: column name to use in x axis. If not provided will use all column
        :param y_cols:
        :param n_samples: samples to take. Take all sample if n_samples=-1
        :return:
        """
        df = self._get_data()

        if len(x_cols) == 0:
            x_cols = list(df)
        if len(y_cols) == 0:
            y_cols = list(df)

        if n_samples > 0:
            df = df.sample(min(n_samples, len(df)))

        # taking 2 different columns and plot their relationship with scatter plot
        for (col1, col2) in tqdm(list(itertools.product(x_cols, y_cols)), desc='Pair plotting'):
            df.plot.scatter(x=col1, y=col2, title=f'Pairplot ({col1} vs {col2})')

            path = self._get_path(name=f'pairplot_{col1}-{col2}', typ='fig')
            pyutils.plt_save(path=path)

    def _stat_missing_data(self, df=None):
        """
        Print percentage of missing data in different columns
        :param df: dataframe to calculate. if None load default one
        :return:
        """
        if df is None:
            df = self._get_data()

        pyutils.header('Missing data', typ=2)
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data)


class DatasetC1(Dataset):
    """ Capital one dataset
    Data loading and processing can be different from dataset to dataset.
    """

    def __init__(self, name: str, path: str):
        """
        :param name: name to better identifying dataset
        :param path: full path of the dataset (Expect the dataset to be in csv)
        """
        super().__init__(name=name, path=path)
        # Columns that only present in original data
        self._org_cols = None

        # process and load data
        self._df = self._process_data()

    def _get_data(self):
        """
        Get numerical data. All object/categorical data are converted to numeric value.
        :return:
        """
        numeric_cols = self.get_numeric_col(self._df)
        return self._df[numeric_cols]

    def _get_full_data(self):
        """
        Get complete dataset
        :return:
        """
        return self._df

    def _get_original_data(self) -> pd.DataFrame:
        """
        Data that are in the original dataset
        :return:
        """
        df = self._get_full_data()
        return df[self._org_cols]

    def _process_data(self):
        """
        Converting all non numeric data into category and store in the class
        :return:
        """
        df = pd.read_csv(self.path)
        # storing original columns
        self._org_cols = list(df)

        df = self.as_category(df=df)
        return df

    def describe(self, col_target='isFraud_ascat_dupli'):
        """
        Full analysis of the dataset
        :return:
        """

        pyutils.pd_set_display()

        df = self._get_data()
        df_full = self._get_data()

        self._structure()
        self._stat_missing_data(self._get_original_data())
        # self._plot_columns()
        # self._plot_corr(y_col=col_target)
        # self._plot_pairplot(x_cols=[col_target])


def main():
    dset = DatasetC1(name='small', path=router.dataset_small)
    # dset = DatasetC1(name='large', path=router.dataset_large)
    dset.describe()


if __name__ == '__main__':
    main()
