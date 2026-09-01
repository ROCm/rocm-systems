#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Regenerate ISA and DBT sources for every supported AMDGPU ISA, then format
# changed generated sources through the active virtual environment.
#
# Usage:
#   ./scripts/generate-amdisa.sh

set -Eeuo pipefail

usage() {
  printf 'usage: %s\n' "$(basename "$0")"
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if (($# == 1)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if (($# != 0)); then
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
rocjitsu="$(cd "$script_dir/.." && pwd -P)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)" \
  || die "could not find repository root from $script_dir"
repo="$(cd "$repo" && pwd -P)"
isa_xml="$repo/shared/machine-readable-isa/isa"
isa_out="$rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated"
dbt_out="$rocjitsu/lib/rocjitsu/src/rocjitsu/code/dbt/generated"

discover_virtualenv() {
  # Keep the generator and pre-commit installation coupled to one active
  # virtual environment. PYTHONPATH below selects amdisa from this checkout.
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    printf '%s\n' "$VIRTUAL_ENV"
    return
  fi

  local active_python active_prefix
  active_python="$(command -v python || true)"
  if [[ -n "$active_python" ]]; then
    active_prefix="$(
      "$active_python" -c \
        'import sys; print(sys.prefix if sys.prefix != sys.base_prefix else "")'
    )"
  fi
  [[ -n "${active_prefix:-}" ]] \
    || die "no active Python virtual environment; activate one before running"
  printf '%s\n' "$active_prefix"
}

venv="$(discover_virtualenv)"
python="$venv/bin/python"
pre_commit="$venv/bin/pre-commit"

# \NPI new ISA family: add its in-tree MR ISA XML to this supported-ISA list.
isa_entries=(
  "cdna1:$isa_xml/amdgpu_isa_cdna1.xml"
  "cdna2:$isa_xml/amdgpu_isa_cdna2.xml"
  "cdna3:$isa_xml/amdgpu_isa_cdna3.xml"
  "cdna4:$isa_xml/amdgpu_isa_cdna4.xml"
  "rdna1:$isa_xml/amdgpu_isa_rdna1.xml"
  "rdna2:$isa_xml/amdgpu_isa_rdna2.xml"
  "rdna3:$isa_xml/amdgpu_isa_rdna3.xml"
  "rdna3_5:$isa_xml/amdgpu_isa_rdna3_5.xml"
  "rdna4:$isa_xml/amdgpu_isa_rdna4.xml"
  "cdna5:$isa_xml/amdgpu_isa_cdna5.xml"
)

for dir in "$repo" "$rocjitsu" "$isa_out" "$dbt_out" "$venv"; do
  [[ -d "$dir" ]] || die "directory not found: $dir"
done
for exe in "$python" "$pre_commit"; do
  [[ -x "$exe" ]] || die "executable not found: $exe"
done
for entry in "${isa_entries[@]}"; do
  [[ -f "${entry#*:}" ]] || die "ISA XML not found: ${entry#*:}"
done

export PATH="$venv/bin:$PATH"
export PYTHONPATH="$rocjitsu/lib/python${PYTHONPATH:+:$PYTHONPATH}"

log "Running amdisa generator"
"$python" -m amdisa \
  "${isa_entries[@]}" \
  --isa-output "$isa_out" \
  --include-root "$rocjitsu/lib/rocjitsu/src" \
  --dbt-output "$dbt_out"

isa_rel="${isa_out#"$repo/"}"
dbt_rel="${dbt_out#"$repo/"}"

log "Formatting generated files"
generated_files="$(
  git -C "$repo" ls-files -mo --exclude-standard -- "$isa_rel" "$dbt_rel"
)"
format_files=()
if [[ -n "$generated_files" ]]; then
  mapfile -t format_files <<<"$generated_files"
fi
if ((${#format_files[@]} > 0)); then
  (
    cd "$repo"
    # pre-commit exits nonzero when clang-format rewrites files. Run it again
    # to distinguish that expected first result from a persistent hook failure.
    if ! "$pre_commit" run clang-format --files "${format_files[@]}"; then
      log "Verifying files updated by clang-format"
      "$pre_commit" run clang-format --files "${format_files[@]}"
    fi
  )
else
  printf '  no changed generated files to format\n'
fi
