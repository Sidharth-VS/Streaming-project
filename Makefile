PYTHON ?= python3

<<<<<<< Updated upstream
.PHONY: setup test experiments run-all clean docker-build docker-up docker-down
=======
.PHONY: setup test experiments run-all clean docker-build docker-up docker-smoke \
        docker-logs docker-results docker-csv docker-down
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
	docker compose up --build
=======
	docker compose up -d --build

# Short end-to-end check: 20 windows (~40 s of simulated time).
docker-smoke:
	MAX_WINDOWS=20 docker compose up -d --build

docker-logs:
	docker compose logs -f

# Recorded results: window table + aggregate JSON summary (results/docker_summary.json)
docker-results:
	$(PYTHON) scripts/summarize_docker_results.py

# Raw per-window CSV written by the root node
docker-csv:
	@cat results/docker_results.csv
>>>>>>> Stashed changes

docker-down:
	docker compose down
