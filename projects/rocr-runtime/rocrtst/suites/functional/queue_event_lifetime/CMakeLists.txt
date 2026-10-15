# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)

# This test can also be built alone against an installed ROCr runtime.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(queue_event_lifetime_test LANGUAGES CXX)
    find_package(hsa-runtime64 REQUIRED CONFIG)
    enable_testing()
endif()

add_executable(queue_event_lifetime_test queue_event_lifetime_test.cpp)
target_compile_features(queue_event_lifetime_test PRIVATE cxx_std_17)
target_link_libraries(
    queue_event_lifetime_test
    PRIVATE hsa-runtime64::hsa-runtime64 dl pthread
)
# Export the ioctl interposer for calls made by the loaded thunk library.
target_link_options(queue_event_lifetime_test PRIVATE -Wl,--export-dynamic)
add_test(NAME queue_event_lifetime_test COMMAND queue_event_lifetime_test)
set_tests_properties(
    queue_event_lifetime_test
    PROPERTIES LABELS "standard;stress" TIMEOUT 60 SKIP_RETURN_CODE 77
)
install(TARGETS queue_event_lifetime_test RUNTIME DESTINATION bin)
