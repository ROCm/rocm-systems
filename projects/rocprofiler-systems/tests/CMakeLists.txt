#
#   rocprofiler-systems tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-testing.cmake)

# test groups
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-config-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pthread-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-user-api-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-mpi-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-kokkos-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-openmp-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-code-coverage-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-fork-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-time-window-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rccl-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-overflow-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-annotate-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal-tests.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-python-tests.cmake)

add_subdirectory(source)
