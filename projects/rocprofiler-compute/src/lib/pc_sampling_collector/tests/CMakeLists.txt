## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test-pc-sampling-collector)

add_executable(
    ${target}
    main.cpp
    test_pc_sampling_collector.h
    test_pc_sampling_collector.cpp
    test_code_object_writer.h
    test_code_object_writer.cpp
    mocks.h
    mocks.cpp
)

target_link_libraries(
    ${target}
    PRIVATE pc-sampling-collector gtest_main gmock gsl_assert fmt::fmt nlohmann_json
)
target_include_directories(${target} PUBLIC .)

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
