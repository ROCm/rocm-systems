#!/usr/bin/env bash
#
################################################################################
##
## Copyright (C) Advanced Micro Devices. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################


# This program runs github actions locally using 'act'
# All arguments given to this script are passed to act itself

set -eu
set -o pipefail

get_remote() {
    # redirect all stderr to /dev/null
    exec 3>&2
    exec 2>/dev/null

    git config --get branch."$(git rev-parse --abbrev-ref HEAD)".remote || echo origin

    # restore stderr
    exec 2>&3
}

REMOTE_NAME="${REMOTE_NAME:-$(get_remote)}"
CONTAINER="${CONTAINER:-ubuntu}"

# act will use this file if it exists for vars
VARS_FILE="${VARS_FILE:-act.secrets}"

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

if test -e "${VARS_FILE}"; then
    args+=(--var-file "${VARS_FILE}")
fi

# trace
set -x

mkdir -p /tmp/artifacts
act "${args[@]}" "$@"
