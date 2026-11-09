# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Usage: rj_add_test_path(<target> <macro-name> <path>)
#   target     - existing executable or library
#   macro-name - runtime path macro for this target's own C++ sources
#   path       - runtime value, including generator expressions
# Repeating the same value is a no-op; conflicting values fail configuration.
#
# Each macro names an extern const char[] with unknown bound. Supported use:
# std::string(CONFIG_DIR) + "/file". Literal concatenation, constexpr string
# contents and sizeof are unsupported. Macros do not propagate to targets
# that link this target.
#
# Keep runtime fixture paths in a small per-target object. The forced header
# preserves the original command-line macros' visibility and availability,
# while all consumers can share cached objects between workspace prefixes.
function(rj_add_test_path target name value)
    if(NOT name MATCHES "^[A-Za-z_][A-Za-z_0-9]*$")
        message(FATAL_ERROR "Invalid test path macro: ${name}")
    endif()
    get_property(defined TARGET ${target} PROPERTY "RJ_TEST_PATH_${name}" SET)
    if(defined)
        get_property(previous TARGET ${target} PROPERTY "RJ_TEST_PATH_${name}")
        if(NOT previous STREQUAL value)
            message(
                FATAL_ERROR
                "Conflicting ${name} paths for ${target}: ${previous} and ${value}"
            )
        endif()
        return()
    endif()
    set_property(TARGET ${target} PROPERTY "RJ_TEST_PATH_${name}" "${value}")

    get_property(
        initialized
        TARGET ${target}
        PROPERTY RJ_TEST_PATHS_INITIALIZED
    )
    string(MAKE_C_IDENTIFIER "${target}" target_id)
    set(namespace "rocjitsu::test::build_paths_${target_id}")
    if(NOT initialized)
        set_property(TARGET ${target} PROPERTY RJ_TEST_PATHS_INITIALIZED TRUE)
        set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}_build_paths")
        set(header "${output_dir}/test_build_paths.h")
        set(source "${output_dir}/$<CONFIG>/test_build_paths.cpp")
        file(
            GENERATE OUTPUT "${header}"
            CONTENT
                "#pragma once\n\nnamespace ${namespace} {\n$<TARGET_PROPERTY:${target},RJ_TEST_PATH_DECLARATIONS>} // namespace ${namespace}\n\n$<TARGET_PROPERTY:${target},RJ_TEST_PATH_ALIASES>"
        )
        file(
            GENERATE OUTPUT "${source}"
            CONTENT
                "#include \"test_build_paths.h\"\n\nnamespace ${namespace} {\n$<TARGET_GENEX_EVAL:${target},$<TARGET_PROPERTY:${target},RJ_TEST_PATH_VALUES>>} // namespace ${namespace}\n"
            TARGET ${target}
        )
        target_include_directories(${target} PRIVATE "${output_dir}")
        if(MSVC)
            target_compile_options(
                ${target}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/FI${header}>"
            )
        else()
            target_compile_options(
                ${target}
                PRIVATE
                    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${header}\">"
            )
        endif()
        target_sources(${target} PRIVATE "${source}")
    endif()

    # Store text rather than a CMake list: C++ declarations contain semicolons.
    get_property(
        declarations
        TARGET ${target}
        PROPERTY RJ_TEST_PATH_DECLARATIONS
    )
    get_property(aliases TARGET ${target} PROPERTY RJ_TEST_PATH_ALIASES)
    get_property(values TARGET ${target} PROPERTY RJ_TEST_PATH_VALUES)
    string(APPEND declarations "extern const char path_${name}[];\n")
    string(APPEND aliases "#define ${name} (::${namespace}::path_${name})\n")
    # Generator expressions can introduce backslashes after configure time.
    # Raw literals preserve those characters; keep the delimiter within C++'s
    # 16-character limit and distinct from literal path contents.
    string(SHA256 path_hash "${value}")
    string(SUBSTRING "${path_hash}" 0 12 path_tag)
    set(delimiter "rj_${path_tag}")
    while(value MATCHES "\\)${delimiter}\"")
        string(SHA256 path_hash "${path_hash}")
        string(SUBSTRING "${path_hash}" 0 12 path_tag)
        set(delimiter "rj_${path_tag}")
    endwhile()
    string(
        APPEND values
        "const char path_${name}[] = R\"${delimiter}(${value})${delimiter}\";\n"
    )
    set_property(
        TARGET ${target}
        PROPERTY RJ_TEST_PATH_DECLARATIONS "${declarations}"
    )
    set_property(TARGET ${target} PROPERTY RJ_TEST_PATH_ALIASES "${aliases}")
    set_property(TARGET ${target} PROPERTY RJ_TEST_PATH_VALUES "${values}")
endfunction()
