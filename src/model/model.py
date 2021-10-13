from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, List

from lightgbm import LGBMClassifier
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.middleware.logger import configure_logger

logger = configure_logger(name=__name__)


class SUGGEST_TYPE(Enum):
    CATEGORICAL = "categorical"
    INT = "int"
    UNIFORM = "uniform"


class SearchParams(BaseModel):
    name: str
    suggest_type: SUGGEST_TYPE
    value_range: Any


class PreprocessPipeline(object):
    def __init__(self):
        self.pipeline: Pipeline = None
        self.define_default()

        logger.info(f"initialize preprocess pipeline: {self.pipeline}")

    def define_default(self):
        self.pipeline = Pipeline(
            [
                ("standard_scaler", StandardScaler()),
            ]
        )

    def define_pipeline(
        self,
        pipeline: Pipeline,
    ):
        self.pipeline = pipeline


class AbstraceEstimator(ABC):
    def __init__(
        self,
        preprocess_pipeline: Pipeline,
    ):
        self.name: str = "an_estimator"
        self.preprocess_pipeline = preprocess_pipeline
        self.pipeline: Pipeline = None
        self.search_params: List[SearchParams] = []

    @abstractmethod
    def define_model(
        self,
        **params,
    ):
        raise NotImplementedError

    @abstractmethod
    def define_default_search_params(self):
        raise NotImplementedError

    @abstractmethod
    def define_search_params(
        self,
        search_params: List[SearchParams],
    ):
        raise NotImplementedError


class RandomForestClassifierPipeline(AbstraceEstimator):
    def __init__(
        self,
        preprocess_pipeline: Pipeline,
    ):
        super().__init__(
            preprocess_pipeline=preprocess_pipeline,
        )
        self.name: str = "random_forest_classifier"
        self.model: RandomForestClassifier = RandomForestClassifier()
        self.pipeline = deepcopy(self.preprocess_pipeline.pipeline)
        self.pipeline.steps.append(("model", self.model))

        self.define_default_search_params()

        logger.info(f"initialize random forest classifier pipeline: {self.pipeline}")

    def define_model(
        self,
        **params,
    ):
        self.model = RandomForestClassifier(**params)
        self.pipeline = deepcopy(self.preprocess_pipeline.pipeline)
        self.pipeline.steps.append(("model", self.model))

    def define_default_search_params(self):
        self.search_params = [
            SearchParams(
                name="criterion",
                suggest_type=SUGGEST_TYPE.CATEGORICAL,
                value_range=["gini", "entropy"],
            ),
            SearchParams(
                name="max_depth",
                suggest_type=SUGGEST_TYPE.INT,
                value_range=(2, 100),
            ),
            SearchParams(
                name="max_leaf_nodes",
                suggest_type=SUGGEST_TYPE.INT,
                value_range=[2, 100],
            ),
        ]

    def define_search_params(
        self,
        search_params: List[SearchParams],
    ):
        self.search_params = search_params
        logger.info(f"new search param: {self.search_params}")


class LightGBMClassifierPipeline(AbstraceEstimator):
    def __init__(
        self,
        preprocess_pipeline: Pipeline,
    ):
        super().__init__(
            preprocess_pipeline=preprocess_pipeline,
        )
        self.name: str = "light_gbm_classifier"
        self.model: LGBMClassifier = LGBMClassifier()
        self.pipeline = deepcopy(self.preprocess_pipeline.pipeline)
        self.pipeline.steps.append(("model", self.model))

        self.define_default_search_params()

        logger.info(f"initialize lightgbm classifier pipeline: {self.pipeline}")

    def define_model(
        self,
        **params,
    ):
        self.model = LGBMClassifier(**params)
        self.pipeline = deepcopy(self.preprocess_pipeline.pipeline)
        self.pipeline.steps.append(("model", self.model))

    def define_default_search_params(self):
        self.search_params = [
            SearchParams(
                name="num_leaves",
                suggest_type=SUGGEST_TYPE.INT,
                value_range=(2, 100),
            ),
            SearchParams(
                name="max_depth",
                suggest_type=SUGGEST_TYPE.INT,
                value_range=(2, 100),
            ),
            SearchParams(
                name="learning_rate",
                suggest_type=SUGGEST_TYPE.UNIFORM,
                value_range=[0.0001, 0.01],
            ),
            SearchParams(
                name="feature_fraction",
                suggest_type=SUGGEST_TYPE.UNIFORM,
                value_range=[0.001, 0.9],
            ),
        ]

    def define_search_params(
        self,
        search_params: List[SearchParams],
    ):
        self.search_params = search_params
        logger.info(f"new search param: {self.search_params}")
