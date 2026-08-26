# cmake/RcclCxxStandard.cmake
#
# Selects the C++ standard RCCL is built with and applies it to the current
# scope.
#
# ROCm 7.14 is the cut-over point: from that release onwards RCCL builds as
# C++20, while older ROCm stays on C++17. The formatting back end follows this
# choice (see cmake/RcclFormat.cmake), so C++17 builds keep using fmtlib.
#
# Sets in the including scope:
#   RCCL_CXX_STANDARD      17 or 20
#   CMAKE_CXX_STANDARD     mirrors RCCL_CXX_STANDARD
#
# Reads ROCM_VERSION when the including project already probed it (the main
# build does, in the root CMakeLists.txt); otherwise the version is read from
# the ROCm installation so standalone builds such as test/host agree.

set(RCCL_CXX20_MIN_ROCM_VERSION 71400)

if(NOT DEFINED ROCM_VERSION)
  if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
  endif()
  # Multi-arch installs put the version file under a 'core' symlink.
  foreach(_rccl_ver_file "${ROCM_PATH}/.info/version" "${ROCM_PATH}/core/.info/version")
    if(EXISTS "${_rccl_ver_file}")
      file(READ "${_rccl_ver_file}" _rccl_ver_string)
      break()
    endif()
  endforeach()
  if(_rccl_ver_string MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    math(EXPR ROCM_VERSION
      "(10000 * ${CMAKE_MATCH_1}) + (100 * ${CMAKE_MATCH_2}) + ${CMAKE_MATCH_3}")
  endif()
endif()

if(DEFINED ROCM_VERSION AND ROCM_VERSION GREATER_EQUAL ${RCCL_CXX20_MIN_ROCM_VERSION})
  set(RCCL_CXX_STANDARD 20)
else()
  set(RCCL_CXX_STANDARD 17)
endif()

if(NOT DEFINED ROCM_VERSION)
  message(WARNING
    "Could not determine ROCm version; defaulting to C++${RCCL_CXX_STANDARD}.")
else()
  message(STATUS
    "RCCL C++ standard: C++${RCCL_CXX_STANDARD} (ROCm version ${ROCM_VERSION}, "
    "C++20 from ${RCCL_CXX20_MIN_ROCM_VERSION})")
endif()

set(CMAKE_CXX_STANDARD          ${RCCL_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)  # Keep -std=c++NN rather than -std=gnu++NN
