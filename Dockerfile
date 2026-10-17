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
# Normalize line endings: a CRLF checkout on Windows would otherwise make the
# kernel look for a interpreter named "bash\r" and every node exits 127.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "approxiot.runtime.node"]
