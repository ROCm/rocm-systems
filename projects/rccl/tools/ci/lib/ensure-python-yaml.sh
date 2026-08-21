#!/usr/bin/env bash
# Ensure `python3` on PATH can `import yaml` for the RCCL / rccl-tests build.
#
# Some CMake steps shell out to python and `import yaml`; when PyYAML is missing,
# configure aborts with "ModuleNotFoundError: No module named 'yaml'". If it is
# not already importable, create a venv (reusing system site-packages) that
# pip-installs PyYAML and put it first on PATH.
#
# MUST be sourced (not executed) so the venv activation persists in the caller.
# Honors RCCL_CI_VENV to override the venv location.

if python3 -c 'import yaml' 2>/dev/null; then
  echo "==> python yaml: already importable ($(command -v python3))"
else
  _venv_dir="${RCCL_CI_VENV:-${RCCL_DEVICE_API_WORKDIR:?RCCL_DEVICE_API_WORKDIR must be set}/.ci-out/pyvenv}"
  if [[ ! -x "${_venv_dir}/bin/python3" ]]; then
    echo "==> python yaml: creating venv at ${_venv_dir}"
    python3 -m venv --system-site-packages "${_venv_dir}"
  fi
  # shellcheck source=/dev/null
  source "${_venv_dir}/bin/activate"
  if ! python3 -c 'import yaml' 2>/dev/null; then
    echo "==> python yaml: pip installing PyYAML into ${_venv_dir}"
    python3 -m pip install --quiet --disable-pip-version-check PyYAML
  fi
  python3 -c 'import yaml, sys; print("==> python yaml:", yaml.__version__, "via", sys.executable)'
  unset _venv_dir
fi
