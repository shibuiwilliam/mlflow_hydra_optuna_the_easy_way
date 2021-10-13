import mlflow
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.model.model import AbstractEstimator


class Evaluation(BaseModel):
    accuracy: float
    precision: float
    recall: float


class Trainer(object):
    def __init__(self):
        pass

    def train_and_evaluate(
        self,
        pipeline: AbstractEstimator,
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        test_data: pd.DataFrame,
        test_target: pd.DataFrame,
        save_file_path: str,
    ) -> Evaluation:
        with mlflow.start_run(run_name=pipeline.name):
            self.train(
                pipeline=pipeline,
                train_data=train_data,
                train_target=train_target,
            )
            evaluation = self.evaluate(
                pipeline=pipeline,
                test_data=test_data,
                test_target=test_target,
            )

            mlflow.log_params(pipeline.params)
            mlflow.log_metrics(
                {
                    "accuracy": evaluation.accuracy,
                    "precision": evaluation.precision,
                    "recall": evaluation.recall,
                }
            )
            pipeline.save(
                save_file_path=save_file_path,
            )
            mlflow.log_param("model", pipeline.name)
            mlflow.log_artifact(save_file_path)
        return evaluation

    def train(
        self,
        pipeline: AbstractEstimator,
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
    ):
        pipeline.pipeline.fit(X=train_data, y=train_target)

    def evaluate(
        self,
        pipeline: AbstractEstimator,
        test_data: pd.DataFrame,
        test_target: pd.DataFrame,
    ) -> Evaluation:
        predict = pipeline.pipeline.predict(test_data)
        accuracy = accuracy_score(test_target, predict)
        precision = precision_score(test_target, predict, average="micro")
        recall = recall_score(test_target, predict, average="micro")
        return Evaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )
