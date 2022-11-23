""" Make requests """
import json
import os
import logging
import sys
import pandas as pd
import numpy as np
import requests


PATH_TO_DATA = os.path.abspath(os.path.join('data',
                                            'heart_cleveland_upload.csv'))
TARGET = 'condition'
LOCALHOST = "127.0.0.1"
PORT = 9000
DOMAIN = f"{LOCALHOST}:{PORT}"
ENDPOINT = "predict"


def setup_logging():
    """ Logger settings """
    logg = logging.getLogger(__name__)
    while logg.handlers:
        logg.handlers.pop()
    handler = logging.StreamHandler(sys.stdout)
    logg.setLevel(logging.INFO)
    logg.addHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.propagate = False
    return logger


if __name__ == "__main__":

    logger = setup_logging()

    logger.info("Reading data")
    data = pd.read_csv(PATH_TO_DATA).drop(columns=TARGET)
    request_features = list(data.columns)

    for i, _ in enumerate(data):
        request_data = [
            x.item() if isinstance(x, np.generic) else x
            for x in data.iloc[i].tolist()
        ]

        logger.info(f"Request data:\n {request_data}")

        response = requests.post(f"http://{DOMAIN}/{ENDPOINT}",
                                 json.dumps(request_data))

        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response data:\n {response.json()}")
