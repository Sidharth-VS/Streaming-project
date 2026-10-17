PYTHON ?= python3

.PHONY: setup test experiments run-all clean docker-build docker-up docker-smoke docker-logs docker-results docker-down

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

# Full containerized topology (broker + 15 node containers + tc-netem latency).
# Detached: the broker keeps running after the 15 nodes finish, so inspect the
# logs/results and then `make docker-down`.
docker-up:
	docker compose up -d --build

# Short end-to-end check: 20 windows (~40 s of simulated time).
docker-smoke:
	MAX_WINDOWS=20 docker compose up -d --build

docker-logs:
	docker compose logs -f

docker-results:
	@cat results/docker_results.csv

docker-down:
	docker compose down
