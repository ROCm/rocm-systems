#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Record an asciinema demo of a mirage/rocjitsu workflow.
#
# WHY: the emulation demos under `emulation/rocjitsu/demos/` each ship three
# files — `<demo>.md` (what it shows), `<demo>.sh` (the exact commands), and
# `<demo>.cast` (the recording). This helper is the single, reproducible way to
# (re)generate the `.cast`: it builds a portable mirage + rocjitsu with
# `mirage-docker-build.sh`, puts that `mirage` on `PATH`, and records the demo
# script with asciinema, overwriting the `.cast` next to the `.sh` by default.
#
# Usage:
#   scripts/record_demo.sh [options] <demo.sh> [output.cast]
#
# Examples:
#   # Build mirage+rocjitsu, record emulation/rocjitsu/demos/rocgdb-quickstart.cast
#   scripts/record_demo.sh ../rocjitsu/demos/rocgdb-quickstart.sh
#
#   # Reuse an already-built prefix (skip the ~minutes-long docker build)
#   scripts/record_demo.sh --no-build --prefix ./build/manylinux \
#       ../rocjitsu/demos/rocgdb-quickstart.sh
#
# Options:
#   --no-build            Skip the docker build; reuse --prefix / $MIRAGE_BIN.
#   --prefix DIR          Build into / read the mirage prefix here
#                         (default: mirage/build/manylinux; also $MIRAGE_PREFIX).
#   --title STR           asciinema recording title (default: the demo name).
#   -h, --help            Show this help.
#
# Environment:
#   MIRAGE_BIN            Explicit `mirage` binary to use (implies --no-build).
#   MIRAGE_PREFIX         Same as --prefix.
#   CARGO_PROFILE         Passed to mirage-docker-build.sh (default: release).
#   ASCIINEMA_COLS/ROWS   Terminal size for the recording (default 100x30).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

no_build=0
prefix="${MIRAGE_PREFIX:-$MIRAGE_DIR/build/manylinux}"
title=""
demo=""
out=""

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) no_build=1; shift ;;
    --prefix) prefix="$2"; shift 2 ;;
    --title) title="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "record_demo: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) if [[ -z "$demo" ]]; then demo="$1"; elif [[ -z "$out" ]]; then out="$1"; else
         echo "record_demo: unexpected argument: $1" >&2; exit 2; fi; shift ;;
  esac
done

[[ -n "$demo" ]] || { echo "record_demo: missing <demo.sh>" >&2; usage >&2; exit 2; }
[[ -f "$demo" ]] || { echo "record_demo: no such demo script: $demo" >&2; exit 1; }
demo="$(cd "$(dirname "$demo")" && pwd)/$(basename "$demo")"
# Default output: <demo>.cast next to the .sh, overwritten.
[[ -n "$out" ]] || out="${demo%.sh}.cast"
[[ -n "$title" ]] || title="$(basename "${demo%.sh}")"

# --- Require asciinema ------------------------------------------------------
# Recording a demo is a developer convenience, which is not a good enough reason
# to mutate the machine it runs on. Provision the tool in the build or container
# environment instead; this script only reports what is missing.
command -v asciinema >/dev/null 2>&1 || {
  echo "record_demo: asciinema not found. Install it and re-run, e.g." >&2
  echo "  apt-get install asciinema  |  dnf install asciinema" >&2
  echo "  pipx install asciinema     |  pip install --user asciinema" >&2
  exit 1
}

# --- Build mirage + rocjitsu (portable) -------------------------------------
mirage_bin="${MIRAGE_BIN:-}"
if [[ -z "$mirage_bin" && "$no_build" -eq 0 ]]; then
  echo "record_demo: building mirage + rocjitsu into $prefix" >&2
  "$SCRIPT_DIR/mirage-docker-build.sh" "$prefix" >&2
fi
if [[ -z "$mirage_bin" ]]; then
  mirage_bin="$prefix/bin/mirage"
fi
[[ -x "$mirage_bin" ]] || {
  echo "record_demo: mirage binary not found at $mirage_bin" >&2
  echo "  build it first (drop --no-build) or set \$MIRAGE_BIN." >&2
  exit 1
}
mirage_bin="$(cd "$(dirname "$mirage_bin")" && pwd)/$(basename "$mirage_bin")"
echo "record_demo: using mirage: $mirage_bin" >&2

# --- Record -----------------------------------------------------------------
# The demo script finds `mirage` on PATH (built above) and may read $MIRAGE_BIN.
# asciinema replays the exact stdout of `bash <demo.sh>`; --overwrite keeps the
# `.cast` regenerable in place.
echo "record_demo: recording $demo -> $out" >&2
env \
  PATH="$(dirname "$mirage_bin"):$PATH" \
  MIRAGE_BIN="$mirage_bin" \
  asciinema rec \
    --overwrite \
    --title "$title" \
    --cols "${ASCIINEMA_COLS:-100}" \
    --rows "${ASCIINEMA_ROWS:-30}" \
    --command "bash '$demo'" \
    "$out"

echo "record_demo: wrote $out" >&2
