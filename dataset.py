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

from sklearn.model_selection import train_test_split

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


def gen_small_dataset(small_dataset_path: str, large_dataset_path: str, sample_percentage=0.10):
    """
    Generate a small dataset from the large one
    :param small_dataset_path: path of the small dataset
    :param large_dataset_path: path of the larger dataset
    :param sample_percentage: how much sample to take from larger dataset
    :return:
    """

    df: pd.DataFrame = pd.read_csv(large_dataset_path)

    nsamples = len(df) * sample_percentage
    small: pd.DataFrame = df.sample(min(nsamples, len(df)))  # taking min in case nsample is larger than dataset size

    small.to_csv(small_dataset_path, index=False)


class Dataset:
    """ Dataset class data load, store, analyze, explore, visualize etc

    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def get_data(self) -> pd.DataFrame:
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

    def _get_target_col_name(self):
        """
        Target column name in the dataset
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
                # converting to category data
                col_converted = df[col].astype('category').cat.codes
                # put a _ before previous column. making it private
                df.rename(columns={col: f'_{col}'}, inplace=True)
                # now col is the converted data
                df[col] = col_converted
        return df

    @staticmethod
    def get_numeric_cols(df):
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

    @staticmethod
    def get_public_cols(df):
        """
        Columns that will mainly used for training or other purposes.
        All columns name not start with a _ is public column
        :param df:
        :return:
        """
        cols = [col for col in list(df) if col[0] != '_']
        return cols

    def _get_path(self, name: str):
        """
        Get dataset specific path
        : param name: file name
        :return:
        """
        path = os.path.join(router.fig, f'{self.name}_{name}')
        return path

    def structure(self, df=None):
        """
        Basic structure of the data, like number of rows, columns, data types, some sample data etc.
        :param df: passed dataframe. if not passed it'll load default data
        :return:
        """
        if df is None:
            df = self.get_data()

        pyutils.header(f'Basic dataset structure ({self.name})')
        print(f'Number of rows: {len(df)} x_cols: {len(list(df))}')
        print(f'Columns name: {list(df)}')
        pyutils.header(line=f'Basic information:', type_=2)
        print(df.info(null_counts=True))
        pyutils.header(line=f'Sample data:', type_=2)
        print(df.head())

    def _plot_columns(self):
        """
        Plot data of different columns
        :return:
        """
        df = self.get_data()
        for col_name in tqdm(self.get_public_cols(df), desc='Plotting columns'):
            col = df[col_name]

            # plot histogram
            col.plot.hist(title=col_name)
            path_dist = self._get_path(name=f'{col_name}_dist')
            pyutils.plt_save(path=path_dist)

            # plot scatter plot
            col.reset_index(name=col_name).plot.scatter(x='index', y=col_name, title=col_name)
            path_scatter = self._get_path(name=f'{col_name}_scatter')
            pyutils.plt_save(path=path_scatter)

    def _plot_corr(self, cols: List[str] = (), y_col=None):
        """
        Plot correlation between different columns.
        :param cols: columns name to use. If empty, use all column in the dataframe
        :param y_col: print value of y_col column. Print nothing if none is passed
        :return:
        """
        print(f'Plotting correlation between columns..', file=sys.stderr)
        # correlation matrix
        df = self.get_data()
        if len(cols) == 0:
            cols = list(df)

        # find correlation
        corrmat = df[cols].corr()

        # dropping any column if nan is present
        # corrmat = corrmat.dropna()

        # plot correlation matrix
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)

        path = self._get_path(name=f'heatmap')
        pyutils.plt_save(path=path)

        # print y_col corelation with others
        if y_col:
            pyutils.header(f'Correlation of ({y_col}) with other columns', type_=2)
            vals = corrmat[y_col].sort_values()
            print(vals)

    def _plot_pairplot_sns(self, cols: List[str] = (), n_samples=100):
        """
        Pair column plot with sns. Too slow
        :param cols:
        :return:
        """
        print(f'Plotting pairplot columns..', file=sys.stderr)

        df = self.get_data()
        if len(cols) == 0:
            cols = list(df)

        df = df[cols]
        df = df.sample(min(n_samples, len(df)))

        sns.set()
        sns.pairplot(df[cols], height=2.5, diag_kind='kde')

        path = self._get_path(name=f'scatter_matrix')
        pyutils.plt_save(path=path)

    def _plot_pairplot(self, x_cols=[], y_cols=[], n_samples=-1):
        """
        Plot column pairwise
        :param x_cols: column name to use in x axis. If not provided will use all column
        :param y_cols:
        :param n_samples: samples to take. Take all sample if n_samples=-1
        :return:
        """
        df = self.get_data()

        if len(x_cols) == 0:
            x_cols = list(df)
        if len(y_cols) == 0:
            y_cols = list(df)

        if n_samples > 0:
            df = df.sample(min(n_samples, len(df)))

        # taking 2 different columns and plot their relationship with scatter plot
        for (col1, col2) in tqdm(list(itertools.product(x_cols, y_cols)), desc='Pair plotting'):
            df.plot.scatter(x=col1, y=col2, title=f'Pairplot ({col1} vs {col2})')

            path = self._get_path(name=f'pairplot_{col1}-{col2}')
            pyutils.plt_save(path=path)

    def describe_null_values(self, df):

        """
        Print percentage of missing data in different columns
        :param df: dataframe to calculate. if None load default one
        :return:
        """
        if df is None:
            df = self.get_data()

        pyutils.header('Missing data', type_=2)
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        pyutils.pd_set_display()
        print(missing_data)

    def describe_features(self, df=None):
        """
        Generate differen stat of columns
        :param df: dataframe to calculate. if None load default one
        :return:
        """
        if df is None:
            df = self.get_data()

        pyutils.header('Features descriptions', type_=2)

        df1 = df.describe(include='all')

        df1.loc['dtype'] = df.dtypes
        df1.loc['size'] = len(df)
        df1.loc['null(%)'] = df.isnull().mean().round(2)
        df1 = df1.round(2).transpose()
        pyutils.pd_set_display()
        print(df1)
        path = os.path.join(router.tmp, self._get_path(name='detail_stat.csv'))
        df1.to_csv(path)

    def save(self, path):
        """
        Save process data with new path
        :param path:
        :return:
        """
        df = self.get_data()
        df.to_csv(path)

    def get_data_as_xy(self):
        """
        Get data as X (feature) and Y (target) splitted
        :return:
        """
        df = self.get_data()
        y_col_name = self._get_target_col_name()
        x_cols_name = list(set(list(df)) - set([y_col_name]))
        X, y = df[x_cols_name], df[y_col_name]
        return X, y

    def get_train_test_split(self, test_size=0.33):
        df = self.get_data()
        y_col_name = self._get_target_col_name()
        x_cols_name = list(set(list(df)) - set([y_col_name]))

        X, y = df[x_cols_name], df[y_col_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test


class DatasetC1(Dataset):
    """ Capital one dataset
    Data loading and processing can be different from dataset to dataset.
    """

    def __init__(self, name: str, path: str, process_data=True):
        """
        :param name: name to better identifying dataset
        :param path: full path of the dataset (Expect the dataset to be in csv)
        :param process_data: if data need processing
        """
        super().__init__(name=name, path=path)
        # target column name
        self._target_col_name = 'isFraud'

        # want to process the data or not
        if process_data:
            # process and load data
            self._df = self._process_data()
        else:
            # assuming data is already processed. Only load them without processing
            self._df = self.load_data(path=self.path)

    def get_data(self):
        """
        Get numerical data. All object/categorical data are converted to numeric value.
        :return:
        """
        numeric_cols = self.get_public_cols(self._df)
        return self._df[numeric_cols]

    def _get_full_data(self):
        """
        Get complete dataset
        :return:
        """
        return self._df

    @staticmethod
    def load_data(path) -> pd.DataFrame:
        """
        Data that are in the original dataset
        :return:
        """
        df = pd.read_csv(path)
        return df

    def _process_data(self):
        """
        Converting all non numeric data into category and store in the class
        :return:
        """
        df = self.load_data(path=self.path)

        # remove all nan columns
        df = df.dropna(axis=1, how='all')

        # filling other nan cell with most frequent value in that column
        df = df.fillna(df.mode().iloc[0])

        # create category from object column
        df = self.as_category(df=df)

        return df

    def _get_target_col_name(self):
        return self._target_col_name

    def describe(self):
        """
        Full analysis of the dataset
        :return:
        """

        pyutils.pd_set_display()

        self.structure()
        self.describe_features()
        self._plot_columns()
        self._plot_corr()
        self._plot_pairplot(x_cols=[self._target_col_name])

    def get_reverse_transaction(self, verbose=True):
        """
        Get all rows of reverse transactions
        :param verbose: print different stat of reverse transactions
        :return:
        """
        col_name = 'transactionType'  # where to look for revere category
        reverse_name = 'REVERSAL'  # identifier for reverse transaction
        df = self.get_data()

        print(f'Total fraud cases: {len(df[df["isFraud"]==True])}/{len(df)}')

        # filter out reverse category data
        df_reverse = df[df[col_name] == reverse_name]

        # summary of number of samples and total dollar amount for reverse category
        reverse_summary = df_reverse.groupby([col_name]).agg({
            'accountNumber': 'count',
            'transactionAmount': 'sum'
        }).rename(columns={'accountNumber': 'Number of samples', 'transactionAmount': 'Total dollar amount'})

        # print different statistics
        if verbose:
            n_fraud_total = len(df[df['isFraud']==True])
            n_fraud_filter = len(df_reverse[df_reverse['isFraud'] == True])

            n_fraud_p = (n_fraud_filter/n_fraud_total)*100

            print(f'Number of reversed transactions: {len(df)} fraud cases: {n_fraud_filter}/{n_fraud_total} ({n_fraud_p:.2f}%)')
            print(reverse_summary)

        return df

    def multi_swipe_transaction_stat(self):
        """
        Find different statistics for multi swipe transactions
        :return:
        """
        df = self.get_data()

        # convert columns to datetime data type
        df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])

        # assuming multi swipe transaction happens in the same merchant with same account number and transaction amount
        multi_swipe_cols = ['merchantName', 'accountNumber', 'transactionAmount']

        # get recent transaction of the same group
        df['prev_transactionDateTime'] = df.sort_values(
            ['merchantName', 'accountNumber', 'transactionAmount', 'transactionDateTime']) \
            .groupby(multi_swipe_cols)['transactionDateTime'].shift(1)

        # sorting by transaction datetime within multi swipe group to get back to back transaction
        df = df.set_index(multi_swipe_cols).sort_values(by='transactionDateTime').sort_index()

        # time gap between two back to back transaction
        df['time_gap'] = df['transactionDateTime'] - df['prev_transactionDateTime']

        # generate statistics for different time groups
        for time_gap in [60 * 1, 60 * 2, 60 * 5, float('inf')]:
            # filter out by time gap
            df_multi_swiped = df[df['time_gap'].dt.total_seconds() <= time_gap].reset_index()

            print(f"Time gaph: {time_gap}")
            print(f"#Back to back transaction's time gap of multi swipe transaction: {len(df_multi_swiped)}")
            print(f'Total dollar amount: {df_multi_swiped["transactionAmount"].sum()}')

            n_fraud_total = len(df[df['isFraud'] == True])
            n_fraud = len(df_multi_swiped[df_multi_swiped['isFraud'] == True])
            n_fraud_p = (n_fraud / n_fraud_total) * 100
            print(f'Fraud cases: {n_fraud}/{n_fraud_total} ({n_fraud_p:.2f}%)')


def main():
    # dset = DatasetC1(name='small', path=router.dataset_small)
    dset = DatasetC1(name='large', path=router.dataset_large, process_data=True)
    dset.describe()


if __name__ == '__main__':
    main()

