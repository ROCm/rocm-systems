# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define an OBJECT library with standard include dirs and flags for
# rocjitsu sub-components. ROCJITSU_INCLUDE_DIR and ROCJITSU_SRC_DIR
# must be set before including this module.
#
# Usage: rj_add_object_library(<name> <sources...>)
function(rj_add_object_library name)
    set(sources ${ARGN})

    if(DEFINED RJ_AMDGPU_ISA_OUTPUT_DIR
       AND CMAKE_CURRENT_SOURCE_DIR MATCHES "/isa/arch/amdgpu/[^/]+$")
        get_filename_component(arch "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
        set(rewritten_sources)
        set(generated_outputs)
        foreach(source IN LISTS sources)
            if(IS_ABSOLUTE "${source}" OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
                list(APPEND rewritten_sources "${source}")
            else()
                set(generated_source "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/${source}")
                get_filename_component(stem "${source}" NAME_WE)
                set(generated_header "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/${stem}.h")
                set_source_files_properties("${generated_source}" PROPERTIES GENERATED TRUE)
                list(APPEND rewritten_sources "${generated_source}")
                list(APPEND generated_outputs "${generated_source}" "${generated_header}")
            endif()
        endforeach()

        list(APPEND generated_outputs
            "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/insts.h"
            "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/machine_insts.h"
            "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/operand_types.h"
            "${RJ_AMDGPU_ISA_OUTPUT_DIR}/${arch}/test_encodings.h")
        set_property(GLOBAL APPEND PROPERTY RJ_AMDGPU_ISA_GENERATED_OUTPUTS
            ${generated_outputs})
        set(sources ${rewritten_sources})
    endif()

    add_library(${name} OBJECT ${sources})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${name} PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${HSA_INCLUDE_DIR})
    if(DEFINED RJ_AMDGPU_ISA_GENERATED_INCLUDE_DIR)
        target_include_directories(${name} PRIVATE ${RJ_AMDGPU_ISA_GENERATED_INCLUDE_DIR})
    endif()
    target_link_libraries(${name} PRIVATE util simdojo)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden)
    endif()
endfunction()
