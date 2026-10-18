PYTHON ?= python3

.PHONY: setup test experiments run-all clean docker-build docker-up docker-smoke \
        docker-watch docker-logs docker-results docker-report docker-down

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

# Reproduces paper experiments (Fig. 6-12 analogues) + the novel controller study
experiments:
	$(PYTHON) -m experiments.run_all

run-all: experiments

clean:
	rm -rf results/*.png results/*.csv results/*.json results/*.md results/*.html
	touch results/.gitkeep

docker-build:
	docker compose build

# Full containerized topology (broker + 15 node containers + tc-netem latency).
# Detached: the broker keeps running after the 15 nodes finish, so inspect the
# logs/results and then `make docker-down`.
docker-up:
	docker compose up -d --build

# Short end-to-end check: 20 windows (~40 s).
docker-smoke:
	MAX_WINDOWS=20 docker compose up -d --build

# Live demo view: the root's formatted per-window table and final dashboard.
docker-watch:
	docker compose logs -f root

docker-logs:
	docker compose logs -f

# Raw root CSV (one row per window).
docker-results:
	@cat results/docker_results.csv

# Formatted stdout report + docker_summary.json + Markdown/HTML artifacts.
docker-report:
	$(PYTHON) scripts/summarize_docker_results.py

docker-down:
	docker compose down
