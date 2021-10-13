from typing import Optional, Union
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


class PreprocessPipeline(object):
    def __init__(self):
        self.pipeline: Pipeline = None
        self.set_default()

    def set_default(self):
        self.pipeline = Pipeline(
            [
                ("standard_scaler", StandardScaler()),
            ]
        )

    def set(
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
        self.random_forest_classifier: RandomForestClassifier = RandomForestClassifier()
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.random_forest_classifier))

    def define_random_forest_classifier(
        self,
        **random_forest_classifier_params,
    ):
        self.random_forest_classifier = RandomForestClassifier(**random_forest_classifier_params)
        self.pipeline = deepcopy(self.__preprocess_pipeline)
        self.pipeline.steps.append(("model", self.random_forest_classifier))
