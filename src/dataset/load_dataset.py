from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_iris

from src.middleware.logger import configure_logger

logger = configure_logger(name=__name__)


@dataclass
class IrisDataset:
    data: pd.DataFrame
    target: pd.DataFrame


def load_iris_dataset() -> IrisDataset:
    logger.info("load iris dataset")
    data = load_iris()
    data_df = pd.DataFrame(data.data, columns=data.feature_names)
    target_df = pd.DataFrame(data.target, columns=["target"])
    return IrisDataset(data=data_df, target=target_df)
