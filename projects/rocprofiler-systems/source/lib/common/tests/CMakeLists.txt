# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    lib-common-tests
    OBJECT
    test_discover_llvm_libdir.cpp
    test_environment.cpp
    test_path.cpp
    test_remove_env.cpp
    test_update_env.cpp
)

target_link_libraries(
    lib-common-tests
    PUBLIC
        rocprofiler-systems-common-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-ghc-filesystem
)
