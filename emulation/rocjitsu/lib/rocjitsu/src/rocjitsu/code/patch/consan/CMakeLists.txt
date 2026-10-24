# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_rocm_path)

add_library(rocjitsu_consan_compiler STATIC input.cpp prepare.cpp lds_race.cpp)
rj_find_rocm_program(AMDCXX amdclang++)
rj_find_rocm_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler)
if(NOT AMDCXX OR NOT CLANG_OFFLOAD_BUNDLER)
    message(
        FATAL_ERROR
        "ConSan callable probes require the ROCm device compiler and bundler"
    )
endif()
include(rj_add_device_kernel)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(KERNEL_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/probes)
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})
set(_probe_images)
set(_probe_inputs)
foreach(
    _mode
    942:64
    950:64
    1100:32
    1100:64
    1201:32
    1201:64
    1250:32
)
    string(REPLACE ":" ";" _parts ${_mode})
    list(GET _parts 0 _gpu)
    list(GET _parts 1 _wave)
    set(_wave_option -mno-wavefrontsize64)
    if(_wave EQUAL 64)
        set(_wave_option -mwavefrontsize64)
    endif()
    set(_name consan_probe_gfx${_gpu}_wave${_wave})
    rj_add_probe_object(probes gfx${_gpu} OUTPUT_NAME ${_name}
        COMPILE_OPTIONS -std=c++20 -mcode-object-version=5 ${_wave_option}
        -DCONSAN_PROBE_WAVE_SIZE=${_wave}
        -I${ROCJITSU_SRC_DIR} -I${ROCJITSU_INCLUDE_DIR}
        -I${PROJECT_SOURCE_DIR}/lib/util/include
        DEPENDS ${ROCJITSU_SRC_DIR}/rocjitsu/code/consan/contract.h
        ${ROCJITSU_SRC_DIR}/rocjitsu/code/patch/device_wave_state.h
    )
    list(
        APPEND _probe_images
        ${_gpu}:${_wave}:${KERNEL_OUTPUT_DIR}/${_name}.hsaco
    )
    list(APPEND _probe_inputs ${KERNEL_OUTPUT_DIR}/${_name}.hsaco)
endforeach()
set(_embedded_probes ${CMAKE_CURRENT_BINARY_DIR}/embedded_probes.cpp)
add_custom_command(
    OUTPUT ${_embedded_probes}
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/embed_probes.py
        --output ${_embedded_probes} ${_probe_images}
    DEPENDS embed_probes.py ${_probe_inputs}
    VERBATIM
)
target_sources(rocjitsu_consan_compiler PRIVATE ${_embedded_probes})
target_link_libraries(
    rocjitsu_consan_compiler
    PUBLIC rocjitsu_consan_image rocjitsu_dbt_internal simdojo_headers
)
rj_configure_target(rocjitsu_consan_compiler PRIVATE_INCLUDES WARNINGS HIDDEN)
