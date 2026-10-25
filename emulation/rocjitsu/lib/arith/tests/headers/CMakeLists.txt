# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile public headers independently as C++17 and, when supported, C23.
include(CheckCSourceCompiles)
include(CMakePushCheckState)

# Probe the features used by the interface, not just recognition of -std=c23.
# This is optional and does not change the project's C language requirements.
set(rj_arith_c23_option "")
foreach(option IN ITEMS -std=c23 -std=c2x)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "${option} -pedantic-errors")
    string(MAKE_C_IDENTIFIER "${option}" option_id)
    check_c_source_compiles(
        [=[
#include <stdint.h>
constexpr int success = 0;
constexpr uint64_t level = 1ULL << 40;
constexpr uint64_t mode = 1ULL << 16;
static uint64_t flags = level | mode;
int main(void) {
  switch (flags != 0) { case success: return 1; default: return 0; }
}
]=]
        rj_arith_has_c23_${option_id}
    )
    cmake_pop_check_state()
    if(rj_arith_has_c23_${option_id})
        set(rj_arith_c23_option "${option}")
        break()
    endif()
endforeach()
set(rj_arith_header_languages cpp)
if(rj_arith_c23_option)
    list(APPEND rj_arith_header_languages c)
else()
    message(
        STATUS
        "rj-arith: skipping optional C23 header checks (constexpr unavailable)"
    )
endif()

file(
    GLOB detail_headers
    CONFIGURE_DEPENDS
    "${rj_arith_source_dir}/include/rj_arith/detail/*.h"
)
foreach(
    header_path
    IN
    ITEMS "${rj_arith_source_dir}/include/rj_arith/rj_arith.h" ${detail_headers}
)
    get_filename_component(header "${header_path}" NAME_WE)
    foreach(extension IN LISTS rj_arith_header_languages)
        if(header STREQUAL "rj_arith")
            set(target "rj_arith_umbrella_header_${extension}_test")
            set(template umbrella.c)
        elseif(header STREQUAL "rj_arith_common")
            set(target "rj_arith_common_header_${extension}_test")
            set(template common.c)
        else()
            set(target "${header}_header_${extension}_test")
            set(template detail.c)
        endif()
        set(source
            "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}"
        )
        configure_file("${template}" "${source}" @ONLY)
        add_library(${target} OBJECT "${source}")
        if(extension STREQUAL "c")
            target_compile_options(${target} PRIVATE "${rj_arith_c23_option}")
        else()
            set_target_properties(
                ${target}
                PROPERTIES
                    CXX_STANDARD 17
                    CXX_STANDARD_REQUIRED ON
                    CXX_EXTENSIONS OFF
            )
        endif()
        rj_configure_target(${target} WARNINGS)
    endforeach()
endforeach()
