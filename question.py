import os
import argparse
import matplotlib.pyplot as plt

import pyutils
import dataset
import model_factory
import router


def q_1():
    """
    Generate answers for question 1
    :return:
    """
    dset = dataset.DatasetC1(name='large_unprocessed', path=router.dataset_large, process_data=False)
    pyutils.header('Structure of the data')
    dset.structure(dset.get_data())
    pyutils.header('Missing value statistics in the column')
    dset.describe_null_values(dset.get_data())
    dset.describe_features(dset.get_data())


def q_2():
    """
    Generate answers and plot for question 2
    :return:
    """
    dset = dataset.DatasetC1(name='large_unprocessed', path=router.dataset_large, process_data=False)
    df = dset.get_data()

    # sample frequency by transaction amount
    def gen_by_freq():
        ax = df['transactionAmount'].plot.hist(title='Histogram of processed amount of the transaction', bins=20)
        ax.set_xlabel('Transaction amount')
        plt.tight_layout()
        # saving the figure
        path = os.path.join(router.fig, 'histogram_transaction_amount')
        plt.savefig(path)
        plt.show()

    # fraud frequency by transaction amount
    def gen_by_fraud_freq():
        fraud = df[df['isFraud']==True]
        ax = fraud['transactionAmount'].plot.hist(title='Histogram of fraud across transaction amount', bins=20)
        ax.set_xlabel('Transaction amount')
        ax.set_ylabel('Fraud frequency')
        plt.tight_layout()
        # saving the figure
        path = os.path.join(router.fig, 'histogram_transaction_amount_fraud')
        plt.savefig(path)
        plt.show()

    #
    gen_by_freq()
    gen_by_fraud_freq()


def q_3():
    """
    Date generate for question 3
    :return:
    """
    dset = dataset.DatasetC1(name='large_unprocessed', path=router.dataset_large, process_data=False)
    dset.get_reverse_transaction(verbose=True)
    dset.multi_swipe_transaction_stat()


def q_4_1():
    """
    Train random forest model and get stat for question 4.1
    :return:
    """
    # process dataset
    dset = dataset.DatasetC1(name='large_processed', path=router.dataset_large, process_data=True)
    # train and test model
    model = model_factory.RandomForestModel(name='random_forest', dataset=dset)
    model.train()


def q_4_2():
    """
    Run model analysis with smaller dataset
    :return:
    """
    # generating small dataset from larger one
    dataset.gen_small_dataset(small_dataset_path=router.dataset_small, large_dataset_path=router.dataset_large,
                              sample_percentage=0.10)

    # process smaller dataset
    dset = dataset.DatasetC1(name='small_processed', path=router.dataset_small, process_data=True)
    # train and test model
    model = model_factory.RandomForestModel(name='random_forest', dataset=dset)
    model.train()


def answer():
    q_1()
    q_2()
    q_3()
    q_4_1()
    q_4_2()


def main():
    answer()


if __name__ == '__main__':
    main()
