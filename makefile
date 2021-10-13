ABSOLUTE_PATH := $(shell pwd)
DOCKERFILE := Dockerfile
DOCKER_COMPOSE := docker-compose.yaml
VERSION := 0.0.0

DOCKER_REPOSITORY := mlflow_hydra_optuna

DIR := $(ABSOLUTE_PATH)
TAG = the_easy_way


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

.PHONY: lint
lint:
	black --check --diff --line-length 120 .

.PHONY: sort
sort:
	isort .

.PHONY: fmt
fmt: sort
	black --line-length 120 .

