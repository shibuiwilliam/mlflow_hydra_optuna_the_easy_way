ABSOLUTE_PATH := $(shell pwd)
DOCKERFILE := Dockerfile
DOCKER_COMPOSE := docker-compose.yaml
VERSION := 0.0.0
DOCKER_REPOSITORY := mlflow_hydra_optuna
TAG = the_easy_way
DOCKER_IMAGE := $(DOCKER_REPOSITORY):$(TAG)

.PHONY: req
req:
	poetry export \
		--without-hashes \
		-f requirements.txt \
		--output requirements.txt

.PHONY: req_dev
req_dev:
	poetry export \
		--dev \
		--without-hashes \
		-f requirements.txt \
		--output requirements.dev.txt

.PHONY: install_req
install_req:
	pip install mlflow

.PHONY: lint
lint:
	black --check --diff --line-length 120 . --exclude outputs

.PHONY: sort
sort:
	isort . --skip outputs

.PHONY: fmt
fmt: sort
	black --line-length 120 . --exclude outputs


.PHONY: build
build: 
	docker build \
		-t $(DOCKER_IMAGE) \
		-f $(DOCKERFILE) \
		.

.PHONY: run
run:
	docker run \
		-it \
		--name $(TAG) \
		-v $(ABSOLUTE_PATH)/hydra:/opt/hydra \
		-v $(ABSOLUTE_PATH)/outputs:/opt/outputs \
		$(DOCKER_IMAGE) \
		python -m src.main

.PHONY: run_in
run_in:
	docker run \
		-it \
		--name $(TAG) \
		-v $(ABSOLUTE_PATH)/hydra:/opt/hydra \
		-v $(ABSOLUTE_PATH)/outputs:/opt/outputs \
		$(DOCKER_IMAGE) \
		bash


.PHONY: stop
stop:
	docker rm -f $(TAG)
