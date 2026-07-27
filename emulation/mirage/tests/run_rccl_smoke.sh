#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source .venv-mi350/bin/activate
cargo run -- run --num-nodes 2 --env ROCM_HOME=$(rocm-sdk path --root) python tests/fixtures/ml/dist_smoke.py
