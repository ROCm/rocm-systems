#
#   omnitrace tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-testing.cmake)

# test groups
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-config-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-instrument-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-pthread-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-rocm-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-user-api-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-mpi-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-kokkos-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-openmp-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-code-coverage-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-fork-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-time-window-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-critical-trace-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-attach-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-rccl-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-causal-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-python-tests.cmake)
