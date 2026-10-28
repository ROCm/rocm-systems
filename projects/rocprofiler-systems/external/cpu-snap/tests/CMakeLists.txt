# Top-level test registry. Each component owns its own subdirectory.

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "cpu_snap tests: GTest not found, skipping")
    return()
endif()

enable_testing()
include(GoogleTest)

add_library(cpu_snap_test_support INTERFACE)
target_include_directories(cpu_snap_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(CPU_SNAP_BUILD_CORE)
    add_subdirectory(core)
endif()
if(CPU_SNAP_BUILD_SAMPLING)
    add_subdirectory(sampling)
endif()
if(CPU_SNAP_BUILD_PROCESSING)
    add_subdirectory(processing)
endif()
# Multi-component tests (e.g. core + sampling reader fixtures).
add_subdirectory(integration)
# cpu-snap-run launch logic (header-only).
add_subdirectory(tools)
