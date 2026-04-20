#!/usr/bin/env bash
# pip-install-from-git.sh — fast pip-install of perfxpert from the
# rocm-systems monorepo git URL.
#
# WHY: `pip install "perfxpert @ git+https://...rocm-systems.git@SHA#subdirectory=..."`
# triggers pip's built-in `git submodule update --init --recursive -q`
# on the cloned tree. rocm-systems declares ~34 top-level submodules
# (mscclpp, perfetto, glog, fmt, gtest, dyninst, sqlite, etc.) that
# are unrelated to perfxpert; pip initialises every one of them before
# handing control to our setup.py. On stock rocm/dev-ubuntu-22.04 that
# step takes 3-6 minutes of wasted network.
#
# WHAT THIS SCRIPT DOES: injects `submodule.active=<perfxpert opencode>`
# into the env via GIT_CONFIG_COUNT, so pip's inherited git subprocess
# initialises ONLY the opencode submodule (the one perfxpert's build
# hook actually needs). All other submodules stay uninitialised.
# Measured: 141s → 0.03s on that specific step against the real
# rocm-systems repo on a fast host.
#
# USAGE:
#   scripts/pip-install-from-git.sh                 # install latest from default branch
#   scripts/pip-install-from-git.sh v0.2.0          # install at a tag
#   scripts/pip-install-from-git.sh <SHA>           # install at a specific commit
#   scripts/pip-install-from-git.sh --extras all    # change [extras]; default [all]
#
# You can also pass any additional pip-install flags after the ref:
#   scripts/pip-install-from-git.sh master --user --no-deps
#
# ENV OVERRIDES:
#   PERFXPERT_GIT_URL         default: https://github.com/ROCm/rocm-systems.git
#   PERFXPERT_GIT_SUBDIR      default: experimental/python/perfxpert
#   PERFXPERT_EXTRAS          default: all (set empty to install without [all])
#
# The wrapper is self-contained — sourcing it is not required, just run it.

set -euo pipefail

DEFAULT_URL="https://github.com/ROCm/rocm-systems.git"
DEFAULT_SUBDIR="experimental/python/perfxpert"
DEFAULT_EXTRAS="all"

REPO_URL="${PERFXPERT_GIT_URL:-${DEFAULT_URL}}"
SUBDIR="${PERFXPERT_GIT_SUBDIR:-${DEFAULT_SUBDIR}}"
EXTRAS="${PERFXPERT_EXTRAS-${DEFAULT_EXTRAS}}"

ref=""
pip_extra_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --extras)
      EXTRAS="${2:-}"
      shift 2
      ;;
    --extras=*)
      EXTRAS="${1#--extras=}"
      shift
      ;;
    -h|--help)
      # Print the block comment at the top of this file.
      sed -n '3,35p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
      exit 0
      ;;
    *)
      if [ -z "${ref}" ] && [[ ! "$1" == -* ]]; then
        ref="$1"
      else
        pip_extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

ref_frag=""
if [ -n "${ref}" ]; then
  ref_frag="@${ref}"
fi

extras_frag=""
if [ -n "${EXTRAS}" ]; then
  extras_frag="[${EXTRAS}]"
fi

pkg_spec="perfxpert${extras_frag} @ git+${REPO_URL}${ref_frag}#subdirectory=${SUBDIR}"

# --- env-var injection: scope git submodule init to the opencode path only.
#
# pip uses `os.environ.copy()` when it spawns git, so the two env vars
# below propagate into `git submodule update --init --recursive -q` and
# `git clone` alike. Documented: git-config(1) — "GIT_CONFIG_COUNT",
# "submodule.active", "submodule.<name>.active".
#
# We scope to the opencode submodule because it's the only one the
# perfxpert setup.py build hook actually needs at install time. All
# other ~33 submodules (mscclpp, perfetto, glog, fmt, gtest, dyninst,
# sqlite, …) are unrelated and get skipped entirely.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="submodule.active"
export GIT_CONFIG_VALUE_0="${SUBDIR}/opencode"

echo "pip-install-from-git: installing ${pkg_spec}" >&2
echo "  GIT_CONFIG_COUNT=${GIT_CONFIG_COUNT}" >&2
echo "  GIT_CONFIG_KEY_0=${GIT_CONFIG_KEY_0}" >&2
echo "  GIT_CONFIG_VALUE_0=${GIT_CONFIG_VALUE_0}" >&2

exec pip install "${pkg_spec}" "${pip_extra_args[@]}"
