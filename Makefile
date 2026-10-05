PYTHON ?= python3

.PHONY: setup test experiments run-all clean docker-build docker-up docker-down

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# Reproduces paper experiments (Fig. 6-12 analogues) + the novel controller study
experiments:
	$(PYTHON) -m experiments.run_all

run-all: experiments

clean:
	rm -rf results/*.png results/*.csv results/*.json
	touch results/.gitkeep

docker-build:
	docker compose build

# Full containerized topology (broker + 15 node containers + tc-netem latency)
docker-up:
	docker compose up --build

docker-down:
	docker compose down
