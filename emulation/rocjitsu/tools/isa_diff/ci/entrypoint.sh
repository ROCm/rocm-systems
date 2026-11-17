#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Container entry: gh token from the mounted secret, git settings, then a command.
set -euo pipefail
if [[ -r /run/secrets/gh_token ]]; then
  GH_TOKEN="$(tr -d '[:space:]' < /run/secrets/gh_token)"
  export GH_TOKEN
fi
# Never start a gc in the big clone; accept the host clone mounted read-only.
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1='*'
case "${1:-}" in
  gh) shift; exec gh "$@" ;;
  shell) exec bash ;;
  run) shift; exec "$@" ;;
  *) exec python3 /opt/isadiff/ci/cli.py "$@" ;;
esac
