""" Make custom transformer """
# pylint: disable=E0401, E0402


import logging
from typing import NoReturn

import pandas as pd
from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin

from ..entities.custom_transformer_params import TransformerParams


logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer class """
    def __init__(self, features) -> NoReturn:
        """ Class ctor """
        self.transform_numerical = RobustScaler()
        self.numerical_features = features.numerical_features

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Fitting transformer for numerical features"""
        self.transform_numerical.fit(data[self.numerical_features])

        return self

    def transform(self, data: pd.DataFrame) -> TransformerParams:
        """ Transform numerical feature """

        return self.transform_numerical.transform(data[
            self.numerical_features])
