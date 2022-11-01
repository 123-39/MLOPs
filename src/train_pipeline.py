""" Training pipline for model """
# pylint: disable=E0401, E0611, E1120


import json
import logging
import sys

import click

from entities.train_pipeline_params import read_training_pipeline_params

from data.make_dataset import (
    read_dataset,
    split_train_val_data,
)

from features.build_features import (
    extract_target,
    drop_target,
    build_feature_transformer,
    make_features
)

from features.custom_transformer import (
    CustomTransformer,
)

from models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    save_model,
    save_transformer,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """ train pipeline """
    training_pipeline_params = read_training_pipeline_params(config_path)

    logger.info('Training pipeline with %s...',
                training_pipeline_params.train_params.model_type)

    data = read_dataset(training_pipeline_params.input_data_path)

    train_df, valid_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    train_target = extract_target(
        train_df, training_pipeline_params.feature_params
    )
    train_df = drop_target(
        train_df, training_pipeline_params.feature_params
    )

    valid_target = extract_target(
        valid_df, training_pipeline_params.feature_params
    )
    valid_df = drop_target(
        valid_df, training_pipeline_params.feature_params
    )

    logger.info('train_df.shape is equal %s', train_df.shape)
    logger.info('val_df.shape is equal %s', valid_df.shape)

    if training_pipeline_params.custom_transformer_params.use_custom_transformer:
        transformer = CustomTransformer(training_pipeline_params.feature_params)
        transformer.fit(
            data
        )
    else:
        transformer = build_feature_transformer(training_pipeline_params.feature_params)
        transformer.fit(train_df)

    save_transformer(transformer, training_pipeline_params.save_transformer)

    train_features = make_features(
        transformer,
        train_df
    )

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    valid_feature = make_features(
        transformer,
        valid_df,
    )

    predict = predict_model(
        model,
        valid_feature,
    )

    metrics = evaluate_model(
        predict,
        valid_target,
    )

    logger.info('Metrics: %s', metrics)

    with open(training_pipeline_params.metric_path, 'w',
              encoding='utf-8') as file_metrics:
        json.dump(metrics, file_metrics)

    save_model(model, training_pipeline_params.save_model)


@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config_log_reg.yaml')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
