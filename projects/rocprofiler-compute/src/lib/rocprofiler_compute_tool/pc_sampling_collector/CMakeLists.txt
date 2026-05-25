## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target pc-sampling-collector)
add_library(
    ${target}
    OBJECT
    pc_sampling_collector.h
    pc_sampling_context.h
    pc_sampling_collector.cpp
    code_object_translator.h
    code_object_translator.cpp
    code_object_writer.h
    code_object_writer.cpp
)

target_include_directories(${target} PUBLIC .)

target_link_libraries(
    ${target}
    PUBLIC
        rocprofiler-sdk::rocprofiler-sdk
        gsl_assert
    PRIVATE
        libdw::libdw
        amd_comgr
        nlohmann_json
        fmt::fmt
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
