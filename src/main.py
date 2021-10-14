from typing import Dict, List, Union
import os
import yaml

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

import hydra
from src.dataset.load_dataset import load_iris_dataset
from src.middleware.logger import configure_logger
from src.model.model import (
    SUGGEST_TYPE,
    LightGBMClassifierPipeline,
    PreprocessPipeline,
    RandomForestClassifierPipeline,
    SearchParams,
)
from src.search.search import DIRECTION, OptunaRunner
from src.train.train import Trainer

logger = configure_logger(name=__name__)


def parse_params(params: DictConfig) -> List[SearchParams]:
    search_params = []
    for param in params:
        if param.suggest_type == SUGGEST_TYPE.CATEGORICAL.value:
            search_params.append(
                SearchParams(
                    name=param.name,
                    suggest_type=SUGGEST_TYPE.CATEGORICAL,
                    value_range=param.value_range,
                )
            )
        elif param.suggest_type == SUGGEST_TYPE.INT.value:
            search_params.append(
                SearchParams(
                    name=param.name,
                    suggest_type=SUGGEST_TYPE.INT,
                    value_range=tuple(param.value_range),
                )
            )
        elif param.suggest_type == SUGGEST_TYPE.UNIFORM.value:
            search_params.append(
                SearchParams(
                    name=param.name,
                    suggest_type=SUGGEST_TYPE.UNIFORM,
                    value_range=tuple(param.value_range),
                )
            )

    logger.info(f"params: {search_params}")
    return search_params


@hydra.main(
    config_path="../hydra",
    config_name="default",
)
def main(cfg: DictConfig):
    logger.info(f"config: {cfg}")
    cwd = os.getcwd()
    logger.info(f"os cwd: {cwd}")

    preprocess_pipeline = PreprocessPipeline()

    random_forest_classifier_pipeline = RandomForestClassifierPipeline(preprocess_pipeline=preprocess_pipeline)
    random_forest_classifier_params = parse_params(params=cfg.random_forest_classifier.parameters)
    random_forest_classifier_pipeline.define_search_params(search_params=random_forest_classifier_params)

    lightgbm_classifier_pipeline = LightGBMClassifierPipeline(preprocess_pipeline=preprocess_pipeline)
    lightgbm_classifier_params = parse_params(params=cfg.lightgbm_classifier.parameters)
    lightgbm_classifier_pipeline.define_search_params(search_params=lightgbm_classifier_params)

    estimators = [random_forest_classifier_pipeline, lightgbm_classifier_pipeline]

    iris_dataset = load_iris_dataset()

    optuna_runner = OptunaRunner(
        data=iris_dataset.data,
        target=iris_dataset.target,
        direction=DIRECTION.MAXIMIZE,
        cv=cfg.optuna.cv,
    )
    results = optuna_runner.optimize(
        estimators=estimators,
        n_trials=cfg.optuna.n_trials,
        n_jobs=cfg.optuna.n_jobs,
    )
    logger.info(f"parameter search results: {results}")

    random_forest_best_params: Dict[str, Union[str, float]] = {}
    lightgbm_best_params: Dict[str, Union[str, float]] = {}
    for result in results:
        if result["estimator"] == random_forest_classifier_pipeline.name:
            random_forest_best_params = result["best_params"]
        elif result["estimator"] == lightgbm_classifier_pipeline.name:
            lightgbm_best_params = result["best_params"]

    random_forest_classifier_pipeline.define_model(**random_forest_best_params)
    lightgbm_classifier_pipeline.define_model(**lightgbm_best_params)

    train_data, test_data, train_target, test_target = train_test_split(
        iris_dataset.data,
        iris_dataset.target,
        test_size=0.3,
        random_state=0,
        stratify=iris_dataset.target,
    )
    trainer = Trainer()

    evaluation = trainer.train_and_evaluate(
        pipeline=random_forest_classifier_pipeline,
        train_data=train_data,
        train_target=train_target,
        test_data=test_data,
        test_target=test_target,
        save_file_path=f"{random_forest_classifier_pipeline.name}.pickle",
    )
    logger.info(f"random forest evaluation result: {evaluation}")

    evaluation = trainer.train_and_evaluate(
        pipeline=lightgbm_classifier_pipeline,
        train_data=train_data,
        train_target=train_target,
        test_data=test_data,
        test_target=test_target,
        save_file_path=f"{lightgbm_classifier_pipeline.name}.pickle",
    )
    logger.info(f"lightgbm evaluation result: {evaluation}")

    random_forest_classifier_pipeline.save_params(
        save_file_path=os.path.join(
            cwd,
            ".hydra",
            f"{random_forest_classifier_pipeline.name}.yaml",
        )
    )
    lightgbm_classifier_pipeline.save_params(
        save_file_path=os.path.join(
            cwd,
            ".hydra",
            f"{lightgbm_classifier_pipeline.name}.yaml",
        )
    )


if __name__ == "__main__":
    main()
