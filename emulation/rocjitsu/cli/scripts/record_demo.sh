#!/usr/bin/env bash
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Record an asciinema demo of a rocjitsu/rocjitsu workflow.
#
# WHY: the emulation demos under `emulation/rocjitsu/demos/` each ship three
# files — `<demo>.md` (what it shows), `<demo>.sh` (the exact commands), and
# `<demo>.cast` (the recording). This helper is the single, reproducible way to
# (re)generate the `.cast`: it builds a portable rocjitsu + rocjitsu with
# `rocjitsu-docker-build.sh`, puts that `rocjitsu` on `PATH`, and records the demo
# script with asciinema, overwriting the `.cast` next to the `.sh` by default.
#
# Usage:
#   scripts/record_demo.sh [options] <demo.sh> [output.cast]
#
# Examples:
#   # Build rocjitsu+rocjitsu, record emulation/rocjitsu/demos/rocgdb-quickstart.cast
#   scripts/record_demo.sh ../rocjitsu/demos/rocgdb-quickstart.sh
#
#   # Reuse an already-built prefix (skip the ~minutes-long docker build)
#   scripts/record_demo.sh --no-build --prefix ./build/manylinux \
#       ../rocjitsu/demos/rocgdb-quickstart.sh
#
# Options:
#   --no-build            Skip the docker build; reuse --prefix / $ROCJITSU_BIN.
#   --prefix DIR          Build into / read the rocjitsu prefix here
#                         (default: rocjitsu/build/manylinux; also $ROCJITSU_PREFIX).
#   --title STR           asciinema recording title (default: the demo name).
#   -h, --help            Show this help.
#
# Environment:
#   ROCJITSU_BIN            Explicit `rocjitsu` binary to use (implies --no-build).
#   ROCJITSU_PREFIX         Same as --prefix.
#   CARGO_PROFILE         Passed to rocjitsu-docker-build.sh (default: release).
#   ASCIINEMA_COLS/ROWS   Terminal size for the recording (default 100x30).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCJITSU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

no_build=0
prefix="${ROCJITSU_PREFIX:-$ROCJITSU_DIR/build/manylinux}"
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

# --- Build rocjitsu + rocjitsu (portable) -------------------------------------
rocjitsu_bin="${ROCJITSU_BIN:-}"
if [[ -z "$rocjitsu_bin" && "$no_build" -eq 0 ]]; then
  echo "record_demo: building rocjitsu + rocjitsu into $prefix" >&2
  "$SCRIPT_DIR/rocjitsu-docker-build.sh" "$prefix" >&2
fi
if [[ -z "$rocjitsu_bin" ]]; then
  rocjitsu_bin="$prefix/bin/rocjitsu"
fi
[[ -x "$rocjitsu_bin" ]] || {
  echo "record_demo: rocjitsu binary not found at $rocjitsu_bin" >&2
  echo "  build it first (drop --no-build) or set \$ROCJITSU_BIN." >&2
  exit 1
}
rocjitsu_bin="$(cd "$(dirname "$rocjitsu_bin")" && pwd)/$(basename "$rocjitsu_bin")"
echo "record_demo: using rocjitsu: $rocjitsu_bin" >&2

# --- Record -----------------------------------------------------------------
# The demo script finds `rocjitsu` on PATH (built above) and may read $ROCJITSU_BIN.
# asciinema replays the exact stdout of `bash <demo.sh>`; --overwrite keeps the
# `.cast` regenerable in place.
echo "record_demo: recording $demo -> $out" >&2
env \
  PATH="$(dirname "$rocjitsu_bin"):$PATH" \
  ROCJITSU_BIN="$rocjitsu_bin" \
  asciinema rec \
    --overwrite \
    --title "$title" \
    --cols "${ASCIINEMA_COLS:-100}" \
    --rows "${ASCIINEMA_ROWS:-30}" \
    --command "bash '$demo'" \
    "$out"

echo "record_demo: wrote $out" >&2
