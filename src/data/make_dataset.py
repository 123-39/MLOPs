""" Subpackage for load data """
# pylint: disable=E0401, E0402


import logging
from typing import Tuple
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from ..entities.split_params import SplittingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_dataset(path: str) -> pd.DataFrame:
    """ Read dataset from csv  """

    logger.info("Loading dataset from %s...", path)

    data = pd.read_csv(path)

    logger.info("Finished")
    logger.info("The dataset size: %s", data.shape)

    return data


def split_train_val_data(data: pd.DataFrame,
                         params: SplittingParams,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split data to train and validation """

    logger.info("Splitting dataset to train and test...")

    train_data, val_data = train_test_split(
        data, test_size=params.val_data, random_state=params.random_state)

    logger.info("Finished")

    return train_data, val_data
