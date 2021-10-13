from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        self.preprocess_pipeline = preprocess_pipeline
