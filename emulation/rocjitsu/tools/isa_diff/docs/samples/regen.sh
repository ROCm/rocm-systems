#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Regenerate the committed samples from their sources.
# usage: regen.sh path/to/rj_isa_diff [LLVM_BIN_DIR]
# Needs an LLVM toolchain (llc, llvm-mc, ld.lld; default /opt/rocm/llvm/bin). Paths inside
# the JSON are relative to a scratch directory, so the samples do not depend on where
# they were produced.
set -euo pipefail
TOOL=$(realpath "$1")
LLVM_BIN=${2:-${RJ_ISA_DIFF_LLVM_BIN:-/opt/rocm/llvm/bin}}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"
cp "$HERE"/codeobject/co_old.s "$HERE"/codeobject/co_new.s "$HERE"/llvm_pair/k3.ll .

# Expected exit codes: 0 pass, 1 local mismatch, 2 inconclusive/partial.
expect() {
  local want=$1
  shift
  local rc=0
  "$@" || rc=$?
  if [ "$rc" != "$want" ]; then
    echo "unexpected exit $rc (want $want): $*" >&2
    exit 1
  fi
}

# codeobject: hand-written gfx950 kernels, assembled and linked by the LLVM plugin.
expect 0 "$TOOL" analyze --old co_old.s --new co_new.s --llvm-bin "$LLVM_BIN" --out co
expect 0 "$TOOL" test --case co --cases 5000 --no-case-log
cp co/case.json co/results.json "$HERE"/codeobject/

# synthetic: a wrong opcode injected into the code-object region, detected and replayed.
expect 0 "$TOOL" inject --case co --region r0 --side new --inst 1 --xor 0x01000000 \
  --label wrong-opcode --out syn
expect 1 "$TOOL" test --case syn --cases 1000
expect 0 "$TOOL" replay --counterexample syn/counterexamples/r0-syn-wrong-opcode-case1.json
cp syn/case.json syn/results.json syn/counterexamples/r0-syn-wrong-opcode-case1.json \
  "$HERE"/synthetic/
{
  head -n 3 syn/logs/r0-syn-wrong-opcode.jsonl
  tail -n 1 syn/logs/r0-syn-wrong-opcode.jsonl
} > "$HERE"/synthetic/case_log_excerpt.jsonl

# llvm_pair: one LLVM IR input compiled twice, the new side with the DAG combiner off.
expect 0 "$TOOL" llvm-pair --input k3.ll --llvm-bin "$LLVM_BIN" --common-args "-O3" \
  --old-args "" --new-args "-combiner-disabled" --out llvm
expect 0 "$TOOL" test --case llvm --cases 2000 --no-case-log
cp llvm/case.json llvm/results.json "$HERE"/llvm_pair/
cp llvm/artifacts/old.s "$HERE"/llvm_pair/k3_old.s
cp llvm/artifacts/new.s "$HERE"/llvm_pair/k3_new.s
echo "samples regenerated in $HERE"
