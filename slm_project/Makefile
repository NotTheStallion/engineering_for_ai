#!/usr/bin/bash -x

all: run

run:
	fastapi dev src/main.py

test:
	pytest test/*.py

.PHONY: test