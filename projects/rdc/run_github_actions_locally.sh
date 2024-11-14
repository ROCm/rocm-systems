#!/usr/bin/env bash

# This program runs github actions locally using 'act'
# All arguments given to this script are passed to act itself

set -eu
set -o pipefail

REMOTE_NAME="${REMOTE_NAME:-origin}"
CONTAINER="${CONTAINER:-ubuntu}"

# act will use this file if it exists for secrets
SECRETS_FILE="${SECRETS_FILE:-act.secrets}"

if ! git remote | grep -q "$REMOTE_NAME"; then
    echo "ERROR: Remote '$REMOTE_NAME' does not exist!"
    echo "Set remote like so:
    REMOTE_NAME=myremote $0"
    exit 1
fi

if ! command -v act > /dev/null; then
    echo "ERROR: 'act' not found!"
    echo "Please install act from the link below:"
    echo "https://github.com/nektos/act"
    exit 1
fi

# build arguments array for act
args=(
    --rm
    --remote-name "$REMOTE_NAME"
    --artifact-server-path /tmp/artifacts
    -P "self-hosted=$CONTAINER"
)

if test -e "${SECRETS_FILE}"; then
    args+=(--secret-file "${SECRETS_FILE}")
fi

# trace
set -x

mkdir -p /tmp/artifacts
act "${args[@]}" "$@"
