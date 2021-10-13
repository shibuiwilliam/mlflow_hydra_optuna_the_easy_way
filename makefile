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
