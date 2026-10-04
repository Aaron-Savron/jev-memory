# Validation

The package has been checked with unit tests, invariant scenarios, a local SQLite benchmark, and an Open-Jev T4 evaluation.

## Local checks

- Node tests: 28/28 passing
- Python server tests: 10/10 passing
- Memory invariants: 200/200 passing across 40 narratives
- 10,000-record in-memory benchmark: p50 39.62 ms, p95 46.81 ms, p99 51.79 ms

The benchmark includes receipt writes and excludes model inference.

## Open-Jev T4 check

The experiment uses the pinned Open-Jev-2B checkpoint and the exact Qwen base revision required by the model release. The synthetic memory ranking set contains 30 cases.

- Baseline head: 17/30 correct
- Tuned scalar head: 25/30 correct
- Tuned high-confidence cases: 24/24 correct at confidence >= 0.90
- Tuned inference: p50 409 ms, p95 433 ms
- 12-candidate HTTP serving smoke test: 200 response, about 731 ms wall time
- Peak GPU allocation: about 3.72 GiB on a 15 GiB T4

The tuning and evaluation data are synthetic fixtures. They validate the protocol, calibration path, and resource envelope. They do not claim task-success performance on a production agent workload.

## Dogfood run

`npm run dogfood` processed four CoFound-style events and produced a three-memory deployment context. It retained the user preference, the verified `DATABASE_URL` failure, and the staging restart procedure; it ignored the unverified assistant claim. A correction replaced `systemd` with `Docker`, and deleting the tool source removed the `DATABASE_URL` memory.

Model references: [Open-Jev model card](https://huggingface.co/ZefanCai/Open-Jev-2B), [Open-Jev source](https://github.com/Zefan-Cai/Open-Jev), and [Modal GPU docs](https://modal.com/docs/guide/gpu).
