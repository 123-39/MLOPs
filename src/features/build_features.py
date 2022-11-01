""" Make feature for train model """

# pylint: disable=E0401, E0402
import sys
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from ..entities.feature_params import FeatureParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def build_categorical_pipeline() -> Pipeline:
    """ Make one hot encoding for categorical features """
    categorical_pipeline = Pipeline([
            ('OHE', OneHotEncoder()),
        ])

    return categorical_pipeline


def extract_target(df_heart: pd.DataFrame, params: FeatureParams) -> pd.Series:
    """ Get target column from dataset """
    return df_heart[params.target_col]


def make_features(transformer: ColumnTransformer,
                  df_heart: pd.DataFrame
                  ) -> pd.DataFrame:
    """ Make transform with input pd.DataFrame """
    return transformer.transform(df_heart)


def drop_target(df_heart: pd.DataFrame,
                params,
                ) -> pd.DataFrame:
    """ Delete target column from pf.DataFrame """
    return df_heart.drop(columns=[params.target_col])


def build_feature_transformer(params) -> ColumnTransformer:
    """ Build ColumnTransformer """
    transformer = ColumnTransformer([
        (
            'categorical_pipeline',
            build_categorical_pipeline(),
            params.categorical_features,
        )
    ])

    return transformer
