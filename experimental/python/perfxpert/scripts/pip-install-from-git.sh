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
#   - Uses the host package manager to install missing OS prerequisites when
#     running as root or when sudo is available.
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
    apt install -y curl git unzip python3-venv python3-pip
    python3 -m venv .venv
  RHEL 9:
    command -v curl >/dev/null || dnf install -y curl
    dnf install -y git unzip python3.11 python3.11-pip
    python3.11 -m venv .venv
  RHEL 10:
    command -v curl >/dev/null || dnf install -y curl
    dnf install -y git unzip python3 python3-pip
    python3 -m venv .venv
  SLES 15:
    zypper install -y curl git unzip python311 python311-pip
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

OS prerequisite handling:
  If curl, git, unzip, or a supported Python/pip pair is missing, the wrapper
  installs the distro packages with apt-get, dnf, or zypper when it is running
  as root or sudo is available. This is the package-manager install path.
  Otherwise it prints the exact package-manager command to run first.

Patched perfxpert-code guarantee:
  The pip build hook bootstraps bun when needed, then builds the bundled
  patched opencode binary from the pinned perfxpert submodule. The wrapper
  exits non-zero if that bundled binary is still missing after install.

Supported Ubuntu 24+ flow:
  apt install -y curl git unzip python3-venv python3-pip
  python3 -m venv .venv
  . .venv/bin/activate
  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"

Prerequisite package examples by distro:
  Ubuntu 22.04 / 24.04:
    apt install -y curl git unzip python3-venv python3-pip
    python3 -m venv .venv
  RHEL 9:
    command -v curl >/dev/null || dnf install -y curl
    dnf install -y git unzip python3.11 python3.11-pip
    python3.11 -m venv .venv
  RHEL 10:
    command -v curl >/dev/null || dnf install -y curl
    dnf install -y git unzip python3 python3-pip
    python3 -m venv .venv
  SLES 15:
    zypper install -y curl git unzip python311 python311-pip
    python3.11 -m venv .venv
    # SLES ships the supported interpreter as python3.11 after installing
    # the python311 packages.

Pin a specific ref, tag, or commit hash:
  REF=develop; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
  REF=v0.2.0; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"
  REF=<SHA>; curl -fsSL "https://raw.githubusercontent.com/ROCm/rocm-systems/${REF}/experimental/python/perfxpert/scripts/pip-install-from-git.sh" | bash -s -- "${REF}"

If you are NOT using this wrapper and `perfxpert-code` later reports that
the bundled opencode binary is missing, reinstall with the OS prerequisites
available so pip can bootstrap bun and build the pinned submodule bundle.
EOF
}

_die() {
  echo "pip-install-from-git: $*" >&2
  exit 2
}

_supported_python_with_pip_available() {
  local candidate
  for candidate in python python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 \
      && "${candidate}" -m pip --version >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

_run_with_privilege() {
  local uid
  uid="${EUID:-$(id -u 2>/dev/null || echo 1)}"
  if [ "${uid}" = "0" ]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  return 127
}

_install_os_prereqs_if_needed() {
  local -a missing
  missing=()
  local need_curl need_git need_unzip need_python
  need_curl=0
  need_git=0
  need_unzip=0
  need_python=0

  if ! command -v curl >/dev/null 2>&1; then
    missing+=("curl")
    need_curl=1
  fi
  if ! command -v git >/dev/null 2>&1; then
    missing+=("git")
    need_git=1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    missing+=("unzip")
    need_unzip=1
  fi

  if ! _supported_python_with_pip_available; then
    missing+=("python3.10+ with pip")
    need_python=1
  fi

  if [ "${#missing[@]}" -eq 0 ]; then
    return 0
  fi

  echo "pip-install-from-git: missing OS prerequisites: ${missing[*]}" >&2
  echo "pip-install-from-git: attempting package-manager install" >&2

  if command -v apt-get >/dev/null 2>&1; then
    local -a apt_packages
    apt_packages=()
    [ "${need_curl}" = "1" ] && apt_packages+=("curl")
    [ "${need_git}" = "1" ] && apt_packages+=("git")
    [ "${need_unzip}" = "1" ] && apt_packages+=("unzip")
    [ "${need_python}" = "1" ] && apt_packages+=("python3-venv" "python3-pip")
    if ! _run_with_privilege apt-get update; then
      _die "failed to run apt-get update. Install prerequisites manually:
$(_print_python_prereqs)"
    fi
    if ! _run_with_privilege apt-get install -y "${apt_packages[@]}"; then
      _die "failed to install prerequisites with apt-get. Install them manually:
$(_print_python_prereqs)"
    fi
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    local version_major
    local -a dnf_packages
    version_major=""
    dnf_packages=()
    if [ -r /etc/os-release ]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      version_major="${VERSION_ID%%.*}"
    fi
    [ "${need_curl}" = "1" ] && dnf_packages+=("curl")
    [ "${need_git}" = "1" ] && dnf_packages+=("git")
    [ "${need_unzip}" = "1" ] && dnf_packages+=("unzip")
    if [ "${version_major}" = "9" ]; then
      [ "${need_python}" = "1" ] && dnf_packages+=("python3.11" "python3.11-pip")
      if ! _run_with_privilege dnf install -y "${dnf_packages[@]}"; then
        _die "failed to install prerequisites with dnf. Install them manually:
$(_print_python_prereqs)"
      fi
    else
      [ "${need_python}" = "1" ] && dnf_packages+=("python3" "python3-pip")
      if ! _run_with_privilege dnf install -y "${dnf_packages[@]}"; then
        _die "failed to install prerequisites with dnf. Install them manually:
$(_print_python_prereqs)"
      fi
    fi
    return 0
  fi

  if command -v zypper >/dev/null 2>&1; then
    local -a zypper_packages
    zypper_packages=()
    [ "${need_curl}" = "1" ] && zypper_packages+=("curl")
    [ "${need_git}" = "1" ] && zypper_packages+=("git")
    [ "${need_unzip}" = "1" ] && zypper_packages+=("unzip")
    [ "${need_python}" = "1" ] && zypper_packages+=("python311" "python311-pip")
    if ! _run_with_privilege zypper --non-interactive install -y "${zypper_packages[@]}"; then
      _die "failed to install prerequisites with zypper. Install them manually:
$(_print_python_prereqs)"
    fi
    return 0
  fi

  _die "no supported package manager found for missing prerequisites. Install them manually:
$(_print_python_prereqs)"
}

for _arg in "$@"; do
  case "${_arg}" in
    -h|--help)
      _print_help
      exit 0
      ;;
  esac
done

_verify_bundled_perfxpert_code() {
  "${_PYTHON}" -c '
from importlib import resources
import os
import sys

path = resources.files("perfxpert") / "_bundled" / "opencode"
if not path.is_file():
    print(f"missing bundled opencode: {path}", file=sys.stderr)
    raise SystemExit(1)
if not os.access(path, os.X_OK):
    print(f"bundled opencode is not executable: {path}", file=sys.stderr)
    raise SystemExit(1)
print(path)
'
}

_install_os_prereqs_if_needed

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

if ! command -v curl >/dev/null 2>&1; then
  {
    cat <<'EOF'
pip-install-from-git: `curl` is required for the GitHub install path and bun bootstrap.
EOF
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

if [[ "${_REPO_URL}" == file://* ]]; then
  # Older pip releases reject the PEP 508 direct-reference form for
  # git+file URLs. Keep the customer HTTPS path on PEP 508, but use pip's
  # VCS URL form for local validation and air-gap mirrors.
  _SPEC="${_VCS_TARGET}#egg=${_PACKAGE}&subdirectory=${_SUBDIRECTORY}"
else
  _SPEC="${_PACKAGE} @ ${_VCS_TARGET}#subdirectory=${_SUBDIRECTORY}"
fi

echo "pip-install-from-git: installing ${_SPEC}" >&2

GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=submodule.active \
GIT_CONFIG_VALUE_0="${_SUBMODULE_SCOPE}" \
  "${_PYTHON}" -m pip install "${_PIP_ARGS[@]}" "${_SPEC}"

if ! _BUNDLED_OPENCODE="$(_verify_bundled_perfxpert_code)"; then
  cat >&2 <<'EOF'
pip-install-from-git: install finished, but the bundled patched opencode binary was not produced.

`perfxpert analyze`, `perfxpert-mcp`, and the Python API may still be installed,
but this wrapper requires `perfxpert-code` to be ready end-to-end.
Check the build output above, then retry the same command.
EOF
  exit 2
fi

echo "pip-install-from-git: bundled patched perfxpert-code ready at ${_BUNDLED_OPENCODE}" >&2
