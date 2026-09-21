# nccl4py (RCCL fork) build orchestration.
#
# This is the ROCm-aware counterpart to the upstream NVIDIA nccl4py CMake file.
# Unlike upstream, RCCL's nccl4py package is CUDA-free: setup.py needs no
# CUDA/HIP headers (cudaStream_t & friends are opaque void* aliases) and
# librccl.so is resolved at runtime via dlopen. The Python deps are backed by
# hip-python (see pyproject.toml), so there is no CUDAToolkit gate here.
#
# Targets (only defined when the parent build sets BUILD_NCCL4PY=ON):
#   nccl4py       - build the wheel into ${CMAKE_BINARY_DIR}/dist
#   nccl4py_dev   - create/refresh a uv-managed dev+test environment (editable)
#
# Both targets shell out to `uv`. They are opt-in and never join the default
# ALL target, so enabling BUILD_NCCL4PY does not slow a normal RCCL build.

find_program(UV_EXECUTABLE uv DOC "Path to uv executable")
if(NOT UV_EXECUTABLE)
  # Keep configure succeeding so the option/wiring stays inspectable even on
  # hosts without uv. The targets below are still defined but will fail at
  # build time with a clear 'uv: command not found' until uv is installed:
  #   https://docs.astral.sh/uv/getting-started/installation/
  message(WARNING
    "BUILD_NCCL4PY=ON but 'uv' was not found on PATH. The nccl4py / nccl4py_dev "
    "targets are defined but will fail to build until uv is installed "
    "(https://docs.astral.sh/uv/getting-started/installation/).")
  set(UV_EXECUTABLE uv)
endif()
set(
  UV_ENV
  UV_PYTHON_PREFERENCE=only-managed
  UV_PYTHON=3.13
)

set(NCCL4PY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NCCL4PY_DIST_DIR "${CMAKE_BINARY_DIR}/dist")

# Dev/test environment: sync the `test` dependency group and install the
# package editable. Framework tensors (torch / cupy) are intentionally NOT
# installed here: on ROCm they must come from a ROCm-specific index/build,
# which is left to the test harness / user environment rather than hard-coded.
add_custom_target(
  nccl4py_dev
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" sync --group test
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" pip install -e .
  WORKING_DIRECTORY "${NCCL4PY_DIR}"
  COMMENT "Setting up nccl4py dev/test environment (editable install + test deps)"
)

add_custom_target(
  nccl4py
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" run python --version
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" build --out-dir "${NCCL4PY_DIST_DIR}" "${NCCL4PY_DIR}"
  WORKING_DIRECTORY "${NCCL4PY_DIR}"
  COMMENT "Building nccl4py wheel into ${NCCL4PY_DIST_DIR}"
)
