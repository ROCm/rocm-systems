#!/usr/bin/env bash
# pip-install-from-git.sh — install perfxpert from the rocm-systems monorepo
# while scoping git submodule init down to the opencode subtree only.
#
# Usage:
#   bash scripts/pip-install-from-git.sh
#   bash scripts/pip-install-from-git.sh v0.2.0
#   bash scripts/pip-install-from-git.sh <SHA> --user
#   bash scripts/pip-install-from-git.sh --extras '' --no-deps
#
# Notes:
#   - Requires `git` because pip shells out to `git clone` for VCS installs.
#   - Requires Python 3 + `python -m pip` in the active environment.
#   - On Ubuntu 24+ and other externally managed Python environments,
#     create and activate a virtual environment first.
#   - Prefers the active `python`, then `python3`, then any other
#     already-installed `python3.10+` interpreter on PATH. It never
#     downloads or installs Python for you.

set -euo pipefail

readonly _DEFAULT_REPO_URL="https://github.com/ROCm/rocm-systems.git"
readonly _SUBDIRECTORY="experimental/python/perfxpert"
readonly _SUBMODULE_SCOPE="experimental/python/perfxpert/opencode"

_print_python_prereqs() {
  cat <<'EOF'
Install the prerequisites first:
  Ubuntu 22.04 / 24.04:
    apt install -y curl git python3-venv python3-pip
    python3 -m venv .venv
  RHEL 9:
    dnf install -y curl git python3.11 python3.11-pip
    python3.11 -m venv .venv
  RHEL 10:
    dnf install -y curl git python3 python3-pip
    python3 -m venv .venv
  SLES 15:
    zypper install -y curl git python311 python311-pip
    python3.11 -m venv .venv
  . .venv/bin/activate
EOF
}

_print_help() {
  cat <<'EOF'
pip-install-from-git.sh — install perfxpert from the rocm-systems monorepo
while scoping git submodule init down to the opencode subtree only.

Usage:
  bash scripts/pip-install-from-git.sh
  bash scripts/pip-install-from-git.sh v0.2.0
  bash scripts/pip-install-from-git.sh <SHA> --user
  bash scripts/pip-install-from-git.sh --extras '' --no-deps

Options:
  --extras <name>   Optional extras to install (default: all).
                    Use `--extras ''` to install the base package only.
  --repo-url <url>  Override the git remote (default: ROCm/rocm-systems).
  -h, --help        Show this help.

Interpreter selection:
  The wrapper prefers the active `python`, then `python3`, then any other
  already-installed `python3.10+` interpreter on PATH. It never downloads
  or installs a different Python runtime.

Supported Ubuntu 24+ flow:
  apt install -y curl git python3-venv python3-pip
  python3 -m venv .venv
  . .venv/bin/activate
  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"

Prerequisite package examples by distro:
  Ubuntu 22.04 / 24.04:
    apt install -y curl git python3-venv python3-pip
    python3 -m venv .venv
  RHEL 9:
    dnf install -y curl git python3.11 python3.11-pip
    python3.11 -m venv .venv
  RHEL 10:
    dnf install -y curl git python3 python3-pip
    python3 -m venv .venv
  SLES 15:
    zypper install -y curl git python311 python311-pip
    python3.11 -m venv .venv
    # SLES ships the supported interpreter as python3.11 after installing
    # the python311 packages.

Pin a specific ref, tag, or commit hash:
  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
  REF=v0.2.0; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
  REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"

If `perfxpert-code` later reports that `opencode` is missing, install bun and
run `perfxpert-code install-patches`, or point `PERFXPERT_OPENCODE_PATH` at an
existing opencode binary:
  curl -fsSL https://bun.sh/install | bash
EOF
}

_die() {
  echo "pip-install-from-git: $*" >&2
  exit 2
}

_PYTHON=""
for _candidate in python python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
  if ! command -v "${_candidate}" >/dev/null 2>&1; then
    continue
  fi
  if "${_candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    _PYTHON="${_candidate}"
    break
  fi
done

if [ -z "${_PYTHON}" ]; then
  {
    echo "pip-install-from-git: Python 3.10+ is required."
    echo
    _print_python_prereqs
  } >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  {
    cat <<'EOF'
pip-install-from-git: `git` is required for the GitHub install path.
EOF
    echo
    _print_python_prereqs
  } >&2
  exit 2
fi

if ! "${_PYTHON}" -m pip --version >/dev/null 2>&1; then
  {
    cat <<'EOF'
pip-install-from-git: `python -m pip` is unavailable in the current environment.
EOF
    echo
    _print_python_prereqs
  } >&2
  exit 2
fi

_IN_VENV="$("${_PYTHON}" -c 'import sys; print("1" if sys.prefix != getattr(sys, "base_prefix", sys.prefix) or hasattr(sys, "real_prefix") else "0")')"
_EXTERNALLY_MANAGED="$("${_PYTHON}" -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED")')"

if [ "${_IN_VENV}" != "1" ] && [ -f "${_EXTERNALLY_MANAGED}" ]; then
  {
    cat <<'EOF'
pip-install-from-git: the current Python is externally managed.
EOF
    echo
    echo "Use a virtual environment first:"
    _print_python_prereqs
    echo '  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"'
  } >&2
  exit 2
fi

_REF=""
_EXTRAS="all"
_REPO_URL="${_DEFAULT_REPO_URL}"
declare -a _PIP_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      _print_help
      exit 0
      ;;
    --extras)
      [ "$#" -ge 2 ] || _die "--extras requires a value"
      _EXTRAS="$2"
      shift 2
      ;;
    --repo-url)
      [ "$#" -ge 2 ] || _die "--repo-url requires a value"
      _REPO_URL="$2"
      shift 2
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        _PIP_ARGS+=("$1")
        shift
      done
      ;;
    -*)
      _PIP_ARGS+=("$1")
      shift
      ;;
    *)
      if [ -z "${_REF}" ]; then
        _REF="$1"
      else
        _PIP_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

_PACKAGE="perfxpert"
if [ -n "${_EXTRAS}" ]; then
  _PACKAGE="${_PACKAGE}[${_EXTRAS}]"
fi

_VCS_TARGET="git+${_REPO_URL}"
if [ -n "${_REF}" ]; then
  _VCS_TARGET="${_VCS_TARGET}@${_REF}"
fi

_SPEC="${_PACKAGE} @ ${_VCS_TARGET}#subdirectory=${_SUBDIRECTORY}"

echo "pip-install-from-git: installing ${_SPEC}" >&2

GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=submodule.active \
GIT_CONFIG_VALUE_0="${_SUBMODULE_SCOPE}" \
  "${_PYTHON}" -m pip install "${_PIP_ARGS[@]}" "${_SPEC}"

if ! command -v bun >/dev/null 2>&1 && ! command -v opencode >/dev/null 2>&1 && [ -z "${PERFXPERT_OPENCODE_PATH:-}" ]; then
  cat >&2 <<'EOF'
pip-install-from-git: install completed, but `perfxpert-code` still needs an opencode binary.

`perfxpert analyze`, `perfxpert-mcp`, and the Python API are ready now.
To enable the interactive TUI, either:
  1. install bun (`curl -fsSL https://bun.sh/install | bash`), then run `perfxpert-code install-patches`, or
  2. install opencode separately and ensure it is on PATH, or
  3. set PERFXPERT_OPENCODE_PATH=/path/to/opencode
EOF
fi
