""" Fit and predict model """
# pylint: disable=E0401, E0402
import pickle
import sys
import logging
from typing import Dict, Union, NoReturn

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from ..entities.train_params import TrainingParams

SklearnClassificationModel = Union[LogisticRegression, RandomForestClassifier]


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: TrainingParams
                ) -> SklearnClassificationModel:
    """ Make training model """
    logger.info('Loading %s model...', train_params.model_type)

    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier()
    else:
        logger.exception('Model is incorrect')
        raise NotImplementedError()

    logger.info('Finished loading.')
    logger.info('Fitting...')

    model.fit(features, target)

    logger.info('Finished fitting.')

    return model


def predict_model(model: SklearnClassificationModel,
                  feature: pd.DataFrame
                  ) -> np.ndarray:
    """ Make predict model """
    logger.info('Model predict...')

    predict = model.predict(feature)

    logger.info('Finished predict.')

    return predict


def evaluate_model(predict: np.ndarray,
                   target: pd.Series
                   ) -> Dict[str, float]:
    """ Make evaluate model """

    logger.info('Calculate metrics...')

    return {
        'acc': accuracy_score(target, predict),
        'f1': f1_score(target, predict, average='macro'),
        'roc_auc': roc_auc_score(target, predict),
    }


def save_model(model: SklearnClassificationModel,
               path: str
               ) -> NoReturn:
    """ Save model to file """
    with open(path, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

    logger.info('Model saved.')


def load_model(path: str):
    """ Load model from path """
    logger.info('Model loading...')
    with open(path, 'rb') as model:
        ans = pickle.load(model)
    model.close()

    return ans


def save_transformer(transformer, path: str):
    """ Save transformer from path """
    with open(path, 'wb') as file:
        pickle.dump(transformer, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    logger.info('Transformer saved.')


def load_transformer(path: str):
    """ Load transformer from path """
    logger.info('Transformer loading...')
    with open(path, 'rb') as transformer:
        ans = pickle.load(transformer)
    transformer.close()

    return ans
