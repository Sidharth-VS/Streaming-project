FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# iproute2 supplies `tc` for the paper's tc-netem link latencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first: source edits do not invalidate the (slow) install.
RUN pip install numpy confluent-kafka

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Tolerate a CRLF checkout (core.autocrlf=true on Windows): a stray \r in the
# shebang makes every node exit 127 with `env: 'bash\r': No such file`.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "approxiot.runtime.node"]
