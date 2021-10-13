from omegaconf import DictConfig
import hydra
from typing import List
from src.middleware.logger import configure_logger
from src.search.search import OptunaRunner, DIRECTION
from src.model.model import (
    SUGGEST_TYPE,
    SearchParams,
    PreprocessPipeline,
    RandomForestClassifierPipeline,
    LightGBMClassifierPipeline,
)
from src.dataset.load_dataset import load_iris_dataset

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
    cwd = hydra.utils.get_original_cwd()
    logger.info(f"cwd: {cwd}")

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
    logger.info(f"results: {results}")


if __name__ == "__main__":
    main()
