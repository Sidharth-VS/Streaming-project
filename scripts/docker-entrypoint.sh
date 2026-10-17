#!/usr/bin/env bash
# Apply the paper's WAN latency (§V-A) to this container's egress and start the
# node.  Per-hop latency is applied once per container:
#   sources -> 20 ms, L1 -> 40 ms, L2 -> 80 ms, root -> none.
# The source->L1 hop is the source container's egress, L1->L2 is L1's egress,
# and L2->root is L2's egress, so three netem qdiscs reproduce the paper's
# 20/40/80 ms links, Kafka client traffic included.
set -euo pipefail

DELAY_MS="${EGRESS_DELAY_MS:-0}"
DEV="${NETEM_DEV:-eth0}"

if [[ "${DELAY_MS}" != "0" && "${DELAY_MS}" != "" ]]; then
    if command -v tc >/dev/null 2>&1; then
        if tc qdisc add dev "${DEV}" root netem delay "${DELAY_MS}ms"; then
            echo "[entrypoint] netem: ${DEV} egress delay ${DELAY_MS}ms" >&2
        else
            echo "[entrypoint] WARNING: could not apply netem (need NET_ADMIN); continuing without emulated latency" >&2
        fi
    else
        echo "[entrypoint] WARNING: tc not installed; continuing without emulated latency" >&2
    fi
fi

exec "$@"
