#!/bin/bash
# Resync this directory from NVIDIA/nccl-extensions python/ and reapply the ROCm edits (rocm-port.patch).
#   resync.sh <nccl-extensions checkout>             copy the upstream subset over this tree, then apply the patch
#   resync.sh --refresh <nccl-extensions checkout>   rewrite rocm-port.patch from this tree (after editing)
# The checkout must be at the commit in UPSTREAM_VERSION (--refresh) or the new one (resync; update
# UPSTREAM_VERSION and contrib/nccl_m2n together).
set -euo pipefail
D=$(dirname "$(readlink -f "$0")")
REFRESH=0
if [[ ${1:-} == --refresh ]]; then REFRESH=1; shift; fi
U=$(readlink -f "${1:?usage: resync.sh [--refresh] <nccl-extensions checkout>}")/python

# Upstream files this port takes: packaging, nccl/_extensions and nccl/m2n, without nccl.ep and the examples.
upstream_files() {
  (cd "$U" && find LICENSE.txt MANIFEST.in pyproject.toml setup.py nccl/_extensions nccl/m2n -type f \
     ! -name '*nccl_ep*' ! -path 'nccl/m2n/examples/*' ! -name '*.so' ! -name '*.cpp' ! -path '*/__pycache__/*' |
   sort)
}

if (( REFRESH )); then
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  upstream_files | while read -r f; do
    mkdir -p "$T/a/$(dirname "$f")" "$T/b/$(dirname "$f")"
    cp "$U/$f" "$T/a/$f"
    [[ -f $D/$f ]] && cp "$D/$f" "$T/b/$f"
  done
  (cd "$T" && diff -ruN a b > "$T/out.patch") || [[ $? == 1 ]]
  sed -E 's/^((---|\+\+\+) [^\t]*)\t.*/\1/' "$T/out.patch" > "$D/rocm-port.patch"   # no timestamps
  echo "rocm-port.patch: $(grep -c '^+++ ' "$D/rocm-port.patch") files"
else
  upstream_files | while read -r f; do
    mkdir -p "$D/$(dirname "$f")"
    cp "$U/$f" "$D/$f"
  done
  patch -p1 --no-backup-if-mismatch -d "$D" < "$D/rocm-port.patch"
  echo "Applied rocm-port.patch; on conflicts fix the files, then run resync.sh --refresh."
fi
