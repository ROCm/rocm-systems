#!/usr/bin/env bash

# Build manpages using ronn

set -eu
set -o pipefail

if ! command -v ronn > /dev/null; then
    echo "ERROR: no ronn found!" >&2
    echo "Please follow installation instructions here:" >&2
    echo "https://github.com/apjanke/ronn-ng" >&2
    exit 1
fi

set -x
ronn ./README.md
