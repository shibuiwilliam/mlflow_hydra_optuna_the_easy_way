# mlflow_hydra_optuna_the_easy_way

The easy way to combine [mlflow](https://mlflow.org/), [hydra](https://hydra.cc/) and [optuna](https://optuna.org/) into one machine learning pipeline.

## Objective

TODO

## Usage

### 1. build docker image to run training jobs

```sh
$ make build
docker build \
    -t mlflow_hydra_optuna:the_easy_way \
    -f Dockerfile \
    .
[+] Building 1.8s (10/10) FINISHED
 => [internal] load build definition from Dockerfile                                                                       0.0s
 => => transferring dockerfile: 37B                                                                                        0.0s
 => [internal] load .dockerignore                                                                                          0.0s
 => => transferring context: 2B                                                                                            0.0s
 => [internal] load metadata for docker.io/library/python:3.9.5-slim                                                       1.7s
 => [1/5] FROM docker.io/library/python:3.9.5-slim@sha256:9828573e6a0b02b6d0ff0bae0716b027aa21cf8e59ac18a76724d216bab7ef0  0.0s
 => [internal] load build context                                                                                          0.0s
 => => transferring context: 17.23kB                                                                                       0.0s
 => CACHED [2/5] WORKDIR /opt                                                                                              0.0s
 => CACHED [3/5] COPY .//requirements.txt /opt/                                                                            0.0s
 => CACHED [4/5] RUN apt-get -y update &&     apt-get -y install     apt-utils     gcc &&     apt-get clean &&     rm -rf  0.0s
 => [5/5] COPY .//src/ /opt/src/                                                                                           0.0s
 => exporting to image                                                                                                     0.0s
 => => exporting layers                                                                                                    0.0s
 => => writing image sha256:256aa71f14b29d5e93f717724534abf0f173522a7f9260b5d0f2051c4607782e                               0.0s
 => => naming to docker.io/library/mlflow_hydra_optuna:the_easy_way                                                        0.0s

Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
```

### 2. run parameter search and training job

the parameters for optuna and hyper parameter search are in `hydra/default.yaml`

```sh
$ cat hydra/default.yaml
optuna:
  cv: 5
  n_trials: 20
  n_jobs: 1
random_forest_classifier:
  parameters:
    - name: criterion
      suggest_type: categorical
      value_range:
        - gini
        - entropy
    - name: max_depth
      suggest_type: int
      value_range:
        - 2
        - 100
    - name: max_leaf_nodes
      suggest_type: int
      value_range:
        - 2
        - 100
lightgbm_classifier:
  parameters:
    - name: num_leaves
      suggest_type: int
      value_range:
        - 2
        - 100
    - name: max_depth
      suggest_type: int
      value_range:
        - 2
        - 100
    - name: learning_rage
      suggest_type: uniform
      value_range:
        - 0.0001
        - 0.01
    - name: feature_fraction
      suggest_type: uniform
      value_range:
        - 0.001
        - 0.9


$ make run
docker run \
	-it \
	--name the_easy_way \
	-v ~/mlflow_hydra_optuna_the_easy_way/hydra:/opt/hydra \
	-v ~/mlflow_hydra_optuna_the_easy_way/outputs:/opt/outputs \
	mlflow_hydra_optuna:the_easy_way \
	python -m src.main
[2021-10-14 00:41:29,804][__main__][INFO] - config: {'optuna': {'cv': 5, 'n_trials': 20, 'n_jobs': 1}, 'random_forest_classifier': {'parameters': [{'name': 'criterion', 'suggest_type': 'categorical', 'value_range': ['gini', 'entropy']}, {'name': 'max_depth', 'suggest_type': 'int', 'value_range': [2, 100]}, {'name': 'max_leaf_nodes', 'suggest_type': 'int', 'value_range': [2, 100]}]}, 'lightgbm_classifier': {'parameters': [{'name': 'num_leaves', 'suggest_type': 'int', 'value_range': [2, 100]}, {'name': 'max_depth', 'suggest_type': 'int', 'value_range': [2, 100]}, {'name': 'learning_rage', 'suggest_type': 'uniform', 'value_range': [0.0001, 0.01]}, {'name': 'feature_fraction', 'suggest_type': 'uniform', 'value_range': [0.001, 0.9]}]}}
[2021-10-14 00:41:29,805][__main__][INFO] - os cwd: /opt/outputs/2021-10-14/00-41-29
[2021-10-14 00:41:29,807][src.model.model][INFO] - initialize preprocess pipeline: Pipeline(steps=[('standard_scaler', StandardScaler())])
[2021-10-14 00:41:29,810][src.model.model][INFO] - initialize random forest classifier pipeline: Pipeline(steps=[('standard_scaler', StandardScaler()),
                ('model', RandomForestClassifier())])
[2021-10-14 00:41:29,812][__main__][INFO] - params: [SearchParams(name='criterion', suggest_type=<SUGGEST_TYPE.CATEGORICAL: 'categorical'>, value_range=['gini', 'entropy']), SearchParams(name='max_depth', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='max_leaf_nodes', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100))]
[2021-10-14 00:41:29,813][src.model.model][INFO] - new search param: [SearchParams(name='criterion', suggest_type=<SUGGEST_TYPE.CATEGORICAL: 'categorical'>, value_range=['gini', 'entropy']), SearchParams(name='max_depth', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='max_leaf_nodes', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100))]
[2021-10-14 00:41:29,817][src.model.model][INFO] - initialize lightgbm classifier pipeline: Pipeline(steps=[('standard_scaler', StandardScaler()),
                ('model', LGBMClassifier())])
[2021-10-14 00:41:29,819][__main__][INFO] - params: [SearchParams(name='num_leaves', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='max_depth', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='learning_rage', suggest_type=<SUGGEST_TYPE.UNIFORM: 'uniform'>, value_range=(0.0001, 0.01)), SearchParams(name='feature_fraction', suggest_type=<SUGGEST_TYPE.UNIFORM: 'uniform'>, value_range=(0.001, 0.9))]
[2021-10-14 00:41:29,820][src.model.model][INFO] - new search param: [SearchParams(name='num_leaves', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='max_depth', suggest_type=<SUGGEST_TYPE.INT: 'int'>, value_range=(2, 100)), SearchParams(name='learning_rage', suggest_type=<SUGGEST_TYPE.UNIFORM: 'uniform'>, value_range=(0.0001, 0.01)), SearchParams(name='feature_fraction', suggest_type=<SUGGEST_TYPE.UNIFORM: 'uniform'>, value_range=(0.001, 0.9))]
[2021-10-14 00:41:29,821][src.dataset.load_dataset][INFO] - load iris dataset
[2021-10-14 00:41:29,824][src.search.search][INFO] - estimator: <src.model.model.RandomForestClassifierPipeline object at 0x7f5776aa5f10>
[I 2021-10-14 00:41:29,825] A new study created in memory with name: random_forest_classifier
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
[I 2021-10-14 00:41:30,519] Trial 0 finished with value: 0.96 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'max_leaf_nodes': 62}. Best is trial 0 with value: 0.96.
2021/10/14 00:41:30 WARNING mlflow.tracking.context.git_context: Failed to import Git (the Git executable is probably not on your PATH), so Git SHA is not available. Error: Failed to initialize: Bad git executable.
The git executable must be specified in one of the following ways:
    - be included in your $PATH
    - be set via $GIT_PYTHON_GIT_EXECUTABLE
    - explicitly set via git.refresh()

All git commands will error until this is rectified.

This initial warning can be silenced or aggravated in the future by setting the
$GIT_PYTHON_REFRESH environment variable. Use one of the following values:
    - quiet|q|silence|s|none|n|0: for no warning or exception
    - warn|w|warning|1: for a printed warning
    - error|e|raise|r|2: for a raised exception

Example:
    export GIT_PYTHON_REFRESH=quiet

/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)


<... long training ...>


[I 2021-10-14 00:41:56,870] Trial 19 finished with value: 0.9466666666666667 and parameters: {'num_leaves': 64, 'max_depth': 17, 'learning_rage': 0.0070407009344824675, 'feature_fraction': 0.4416643843187271}. Best is trial 0 with value: 0.9466666666666667.
[2021-10-14 00:41:57,031][src.search.search][INFO] - result for light_gbm_classifier: {'estimator': 'light_gbm_classifier', 'best_score': 0.9466666666666667, 'best_params': {'num_leaves': 17, 'max_depth': 20, 'learning_rage': 0.006952391958964706, 'feature_fraction': 0.8414032025653786}}
[2021-10-14 00:41:57,032][__main__][INFO] - parameter search results: [{'estimator': 'random_forest_classifier', 'best_score': 0.9666666666666668, 'best_params': {'criterion': 'entropy', 'max_depth': 14, 'max_leaf_nodes': 65}}, {'estimator': 'light_gbm_classifier', 'best_score': 0.9466666666666667, 'best_params': {'num_leaves': 17, 'max_depth': 20, 'learning_rage': 0.006952391958964706, 'feature_fraction': 0.8414032025653786}}]
/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py:394: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  self._final_estimator.fit(Xt, y, **fit_params_last_step)
[2021-10-14 00:41:57,518][__main__][INFO] - random forest evaluation result: accuracy=0.9777777777777777 precision=0.9777777777777777 recall=0.9777777777777777
/usr/local/lib/python3.9/site-packages/sklearn/preprocessing/_label.py:98: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/usr/local/lib/python3.9/site-packages/sklearn/preprocessing/_label.py:133: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
[LightGBM] [Warning] Unknown parameter: learning_rage
[LightGBM] [Warning] feature_fraction is set=0.8414032025653786, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.8414032025653786
[2021-10-14 00:41:57,818][__main__][INFO] - lightgbm evaluation result: accuracy=0.9555555555555556 precision=0.9555555555555556 recall=0.9555555555555556
```

### 3. training history and artifacts

training history and artifacts are recorded under `outputs`

```sh
$ tree -a outputs
outputs
├── .gitignore
├── .gitkeep
└── 2021-10-14
    └── 00-41-29
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   ├── light_gbm_classifier.yaml
        │   ├── overrides.yaml
        │   └── random_forest_classifier.yaml
        ├── light_gbm_classifier.pickle
        ├── main.log
        ├── mlruns
        │   ├── .trash
        │   └── 0
        │       ├── 001f4913ee2c464e9095894c280a827f
        │       │   ├── artifacts
        │       │   ├── meta.yaml
        │       │   ├── metrics
        │       │   │   └── accuracy
        │       │   ├── params
        │       │   │   ├── feature_fraction
        │       │   │   ├── learning_rage
        │       │   │   ├── max_depth
        │       │   │   ├── model
        │       │   │   └── num_leaves
        │       │   └── tags
        │       │       ├── mlflow.runName
        │       │       ├── mlflow.source.name
        │       │       ├── mlflow.source.type
        │       │       └── mlflow.user

<... many files ...>

        │       └── meta.yaml
        └── random_forest_classifier.pickle
```

you can also open `mlflow ui`

```sh
$ cd outputs/2021-10-13/13-27-41
$ mlflow ui
[2021-10-13 22:34:51 +0900] [48165] [INFO] Starting gunicorn 20.1.0
[2021-10-13 22:34:51 +0900] [48165] [INFO] Listening at: http://127.0.0.1:5000 (48165)
[2021-10-13 22:34:51 +0900] [48165] [INFO] Using worker: sync
[2021-10-13 22:34:51 +0900] [48166] [INFO] Booting worker with pid: 48166
```

open localhost:5000 in your web-browser

![0](images/0.png)

![1](images/1.png)
