"""ApproxIoT replication (Wen et al., arXiv:1805.05674) + adaptive budget controller.

Package layout mirrors the paper's system decomposition:
- ``sampling``         : reservoir sampling, stratification, weighted hierarchical sampling (Algorithm 2)
- ``error_estimation`` : variance / error-bound estimation (Eq. 10-14, 68-95-99.7 rule)
- ``node``             : per-node workflow (Algorithm 1): sampling nodes and the root
- ``baselines``        : simple random sampling (SRS) baseline
- ``datagen``          : synthetic + real-world stream generators
- ``kafka_io``         : transport abstraction (in-memory sim bus / optional Kafka)
- ``controller``       : NOVEL adaptive sampling-budget controller
- ``metrics``          : throughput / accuracy / latency / bandwidth collection
- ``simulation``       : 8 -> 4 -> 2 -> 1 topology driver used by the experiments
"""

__version__ = "0.1.0"
