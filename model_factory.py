from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import os
import sys
import pickle
import time
import argparse

import router
import dataset


class Model:
    """
    Model abstract class to make different models. Override get_model_skeleton to subclasses.
    """

    def __init__(self, name, dataset, load_offline_model=False):
        self.name = name
        self.dataset = dataset

        self._info = {}  # putting misc. info

        self._model = None  # store model

        # load previously saved model
        if load_offline_model:
            self._load_model()

    def _get_model_path(self):
        """
        Get unique path for model to save
        :return:
        """
        path = os.path.join(router.model, f'{self.name}.model')
        return path

    def _save(self):
        """
        Save model for offline use
        :return:
        """
        path = self._get_model_path()
        with open(path, 'wb') as f:
            pickle.dump(self._get_model(), f)

    def _get_model(self):
        """
        Get current model
        :return:
        """
        return self._model

    def _set_model(self, model):
        """
        Set load model to _model
        :param model:
        :return:
        """
        self._model = model

    def _load_model(self):
        """
        Load model from saved path
        :return:
        """
        path = self._get_model_path()
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self._set_model(model=model)

    def cross_validation(self, K_cv=10):
        """
        Cross validation
        :param K_cv: K of cross validation
        :return:
        """
        rfc_cv = RandomForestClassifier(n_estimators=100)
        X, y = self.dataset.get_data_as_xy()
        scores = cross_val_score(rfc_cv, X, y, cv=K_cv, scoring="accuracy")
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard Deviation:", scores.std())

    def get_model_skeleton(self):
        """
        Get raw model. This needs to be override by sub class
        :return:
        """
        raise NotImplementedError

    def train(self, save=True):
        """
        Train the model
        :param save: if want to save model after training
        :return:
        """
        X_train, X_test, y_train, y_test = self.dataset.get_train_test_split()

        model = self.get_model_skeleton()
        training_start = time.perf_counter()

        # training models and find accuracy and time stat
        model.fit(X_train, y_train)
        training_end = time.perf_counter()
        prediction_start = time.perf_counter()
        preds = model.predict(X_test)
        prediction_end = time.perf_counter()
        acc_rfc = (preds == y_test).sum().astype(float) / len(preds) * 100
        rfc_train_time = training_end - training_start
        rfc_prediction_time = prediction_end - prediction_start
        print(f"Model: {self.name}")
        print(f"Prediction accuracy is: {acc_rfc:.2f}")
        print(f"Time for training: {rfc_train_time:.2f} seconds")
        print(f"Time for prediction: {rfc_prediction_time:.2f} seconds")

        self._info['acc_pred'] = acc_rfc
        self._info['train_time'] = rfc_train_time

        self._set_model(model=model)
        if save:
            self._save()


class KNNModel(Model):

    def __init__(self, name, dataset, load_offline_model=False):
        super().__init__(name=name, dataset=dataset, load_offline_model=load_offline_model)

    def get_model_skeleton(self):
        model = KNeighborsClassifier()
        return model


class RandomForestModel(Model):

    def __init__(self, name, dataset, load_offline_model=False):
        super().__init__(name=name, dataset=dataset, load_offline_model=load_offline_model)

    def get_model_skeleton(self):
        model = RandomForestClassifier(n_estimators=10)
        return model


class NaiveBayesModel(Model):

    def __init__(self, name, dataset, load_offline_model=False):
        super().__init__(name=name, dataset=dataset, load_offline_model=load_offline_model)

    def get_model_skeleton(self):
        model = GaussianNB()
        return model


class SVMModel(Model):

    def __init__(self, name, dataset, load_offline_model=False):
        super().__init__(name=name, dataset=dataset, load_offline_model=load_offline_model)

    def get_model_skeleton(self):
        model = SVC()
        return model

def run_all_models():
    """
    Run all available models
    :return:
    """
    dset = dataset.DatasetC1(name='large', path=router.dataset_large, process_data=True)

    models = dict()

    models['random_forest'] = RandomForestModel(name='random_forest', dataset=dset, load_offline_model=False)
    models['knn'] = KNNModel(name='knn', dataset=dset, load_offline_model=False)
    models['NaiveBayes'] = NaiveBayesModel(name='NaiveBayes', dataset=dset, load_offline_model=False)
    models['SVM'] = SVMModel(name='SVM', dataset=dset, load_offline_model=False)

    for name, model in models.items():
        model.train(save=False)


def run_random_forest():
    """
    Train and do cross validation with random forest model
    :return:
    """
    dset = dataset.DatasetC1(name='large', path=router.dataset_large, process_data=True)
    model = RandomForestModel(name='random_forest', dataset=dset, load_offline_model=False)
    model.train(save=False)
    model.cross_validation(K_cv=10)


def main():
    run_random_forest()
    run_all_models()


if __name__ == '__main__':
    main()
