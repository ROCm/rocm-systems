## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test-rocprofiler-compute-tool)

add_executable(
    ${target}
    test_rocprofiler_compute_tool.h
    test_rocprofiler_compute_tool.cpp
    test_sdk_callbacks.h
    test_sdk_callbacks.cpp
    mocks.h
    mocks.cpp
    main.cpp
)

target_link_libraries(
    ${target}
    PRIVATE rocprofiler-compute-tool gtest_main gmock gsl_assert fmt::fmt
)

if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    # ASAN with -shared-libsan requires explicit pthread linkage (rocm-systems#6455)
    if(THEROCK_SANITIZER MATCHES "ASAN" OR ENABLE_ADDRESS_SANITIZER)
        target_link_options(${target} PRIVATE -lpthread)
    endif()
endif()

set_target_properties(
    ${target}
    PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_PROJECT_NAME}"
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
