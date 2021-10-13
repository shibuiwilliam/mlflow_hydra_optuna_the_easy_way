from typing import Dict, Optional

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.middleware.logger import configure_logger
from src.model.model import LightGBMClassifierPipeline, PreprocessPipeline, RandomForestClassifierPipeline

logger = configure_logger(name=__name__)


class OptunaRunner:
    def __init__(
        self,
        data: pd.DataFrame,
        target: pd.DataFrame,
        cv: int = 5,
        scorings: Dict[str, str] = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        },
    ):
        self.data = data
        self.target = target
        self.cv = cv
        self.scorings = scorings

        optuna.logging.enable_default_handler()

    def _optimize(
        self,
        jobs: Optional[Dict[str, Callable]] = None,
        n_trials: int = 10,
        n_jobs: int = 1,
    ) -> Iterator[Dict[str, Union[str, float]]]:
        if jobs is None:
            jobs = {
                "LogisticRegression": self.optimize_logistic_regression,
                "SVC": self.optimize_svc,
                "DecisionTree": self.optimize_decision_tree,
                "RandomForest": self.optimize_random_forest,
            }

        for model, job in tqdm(jobs.items()):
            study = optuna.create_study()
            study.optimize(job, n_jobs=n_jobs, n_trials=n_trials)
            result = {"model": model, "best_score": 1 - study.best_value, "best_params": study.best_params}
            yield result

    def optimize(
        self, jobs: Optional[Dict[str, Callable]] = None, n_trials: int = 10, n_jobs: int = 1
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Args:
            jobs:
            n_trials:
            n_jobs:
        Returns:
        """
        return list(self._optimize(jobs=jobs, n_jobs=n_jobs, n_trials=n_trials))

    def optimize_logistic_regression(self, trial: optuna.Trial) -> float:

        params = {
            "C": trial.suggest_uniform("C", 0.1, 10),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "intercept_scaling": trial.suggest_uniform("intercept_scaling", 0.1, 2),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "saga"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "multi_class": trial.suggest_categorical("multi_class", ["auto"]),
        }

        clf = self._add_model(LogisticRegression(**params))
        score = cross_validate(
            estimator=clf,
            x=self.data,
            y=self.target,
            cv=self.cv,
            scoring=self.scoring,
            error_score=np.nan,
        )
        return 1 - score.mean()
