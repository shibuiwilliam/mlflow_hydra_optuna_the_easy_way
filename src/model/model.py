from copy import deepcopy
from typing import Optional, Union

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.middleware.logger import configure_logger

logger = configure_logger(name=__name__)


class PreprocessPipeline(object):
    def __init__(self):
        self.pipeline: Pipeline = None
        self.set_default()

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


class RandomForestClassifierPipeline(object):
    def __init__(
        self,
        preprocess_pipeline: Pipeline,
    ):
        self.__preprocess_pipeline = preprocess_pipeline
        self.model: RandomForestClassifier = RandomForestClassifier()
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.model))

    def define_model(
        self,
        **params,
    ):
        self.model = RandomForestClassifier(**params)
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.model))


class LightGBMClassifierPipeline(object):
    def __init__(
        self,
        preprocess_pipeline: Pipeline,
    ):
        self.__preprocess_pipeline = preprocess_pipeline
        self.model: LGBMClassifier = LGBMClassifier()
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.model))

    def define_model(
        self,
        **params,
    ):
        self.model = RandomForestClassifier(**params)
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.model))
