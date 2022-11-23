"""Main part"""
import pickle
import os
import logging
import sys
from typing import NoReturn, List
import uvicorn
import pandas as pd

from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline
from src.response import PredictResponse, InputDataRequest


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
    logg.propagate = False
    return logg


MODEL = None
app = FastAPI()
logger = setup_logging()


def load_pickle(path: str) -> Pipeline:
    """ Read model """

    with open(path, 'rb') as some_object:
        return pickle.load(some_object)


def make_predict(
    data: List,
    features: List[str],
    model: Pipeline,
) -> List[PredictResponse]:
    """ Make predict """
    data = pd.DataFrame(data, columns=features)

    n_row = [i for i, _ in enumerate(data)]
    preds = model.predict(data)

    return [
        PredictResponse(id=index,
                        target=target) for index, target in zip(n_row, preds)
    ]


@app.get('/')
def start() -> str:
    """ Start service info """
    return 'ML service is starting!'


@app.on_event('startup')
def load_model() -> NoReturn:
    """ Model loading """

    global MODEL

    model_path = os.getenv(
      'PATH_TO_MODEL', default='models/log_reg.pkl',
    )

    logger.info('Model from path %s loading...', model_path)

    MODEL = load_pickle(model_path)


@app.get('/health')
def health() -> bool:
    """ Status of model """
    logger.info('Checking health model')
    return not (MODEL is None)


@app.get('/predict')
def predict(request: InputDataRequest) -> List[PredictResponse]:
    """ Model predict """
    if not health():
        logger.error('Model is not health!')
        raise HTTPException(status_code=404, detail='Model not found')

    return make_predict(request.data, request.features, MODEL)


if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=os.getenv('PORT', 9000))
