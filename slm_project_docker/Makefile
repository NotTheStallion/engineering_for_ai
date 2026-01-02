#!/usr/bin/bash -x

all: run

run:
	fastapi dev src/main.py

test:
	pytest test/*.py

docker:
	sudo docker build -t custom-img . 
#--no-cache

.PHONY: test