cmake_minimum_required(VERSION 3.16)
project(hip-trace-lite VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HTL_BUILD_TOOLS "Build htl_dump decoder" ON)
option(HTL_BUILD_TESTS "Build unit + smoke tests" ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# The shared library — populated in later tasks.
add_library(hiptracelite SHARED
    src/htl_loader.cpp
    src/htl_callback.cpp
    src/htl_writer.cpp
)
target_include_directories(hiptracelite PRIVATE src)
target_link_libraries(hiptracelite PRIVATE dl pthread)
set_target_properties(hiptracelite PROPERTIES
    OUTPUT_NAME hiptracelite
    SOVERSION 0)

if(HTL_BUILD_TOOLS)
    add_executable(htl_dump tools/htl_dump.cpp)
    target_include_directories(htl_dump PRIVATE src)
endif()

if(HTL_BUILD_TESTS)
    enable_testing()
    add_executable(test_ring test/test_ring.cpp)
    target_include_directories(test_ring PRIVATE src)
    target_link_libraries(test_ring PRIVATE pthread)
    add_test(NAME ring COMMAND test_ring)

    add_executable(test_record test/test_record.cpp)
    target_include_directories(test_record PRIVATE src)
    add_test(NAME record COMMAND test_record)

    find_package(hip QUIET CONFIG PATHS /opt/rocm)
    if(hip_FOUND)
        add_executable(smoke test/smoke.cpp)
        target_include_directories(smoke PRIVATE src)
        target_link_libraries(smoke PRIVATE hip::host dl pthread)
        # smoke is run manually with the LD_PRELOAD recipe — no add_test() entry.
    else()
        message(STATUS "hip-trace-lite: HIP not found; skipping smoke binary")
    endif()
endif()
