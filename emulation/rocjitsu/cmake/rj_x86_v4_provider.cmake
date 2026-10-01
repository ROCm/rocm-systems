# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Return one digest for a single-config generator, or a configuration-selected
# digest expression for a multi-config generator. Hash only the active flags,
# including user-defined configurations rather than assuming Debug/Release.
function(rj_x86_v4_build_digest output)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "INPUT" "FILES")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ARG_FILES})
    set(_input "${ARG_INPUT}")
    foreach(_file IN LISTS ARG_FILES)
        file(SHA256 "${_file}" _hash)
        string(APPEND _input "|${_file}:${_hash}")
    endforeach()
    if(CMAKE_CONFIGURATION_TYPES)
        set(_result "")
        foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
            string(TOUPPER "${_config}" _upper_config)
            string(
                SHA256 _digest
                "${_input}|${_config}|${CMAKE_CXX_FLAGS}|${CMAKE_CXX_FLAGS_${_upper_config}}"
            )
            string(APPEND _result "$<$<CONFIG:${_config}>:${_digest}>")
        endforeach()
    else()
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _upper_config)
        string(
            SHA256 _result
            "${_input}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_FLAGS}|${CMAKE_CXX_FLAGS_${_upper_config}}"
        )
    endif()
    set(${output} "${_result}" PARENT_SCOPE)
endfunction()

# Flatten COMDAT groups and localize the provider without sharing intermediate
# objects across configurations. Keep both outputs configuration-specific so
# Ninja Multi-Config evaluates object inputs in the requested output config.
function(rj_add_x86_v4_local_object target basename output)
    cmake_parse_arguments(
        PARSE_ARGV
        3
        ARG
        ""
        ""
        "OBJECT_TARGETS;OBJCOPY_OPTIONS"
    )
    set(_reloc "${PROJECT_BINARY_DIR}/${basename}.$<CONFIG>.reloc.o")
    set(_local "${PROJECT_BINARY_DIR}/${basename}.$<CONFIG>.local.o")
    set(_objects "")
    foreach(_target IN LISTS ARG_OBJECT_TARGETS)
        list(APPEND _objects "$<TARGET_OBJECTS:${_target}>")
    endforeach()
    add_custom_command(
        OUTPUT "${_local}"
        BYPRODUCTS "${_reloc}"
        COMMAND
            ${CMAKE_LINKER} -r --force-group-allocation -o "${_reloc}"
            ${_objects}
        COMMAND ${CMAKE_OBJCOPY} ${ARG_OBJCOPY_OPTIONS} "${_reloc}" "${_local}"
        DEPENDS ${ARG_OBJECT_TARGETS} ${_objects}
        COMMENT "Isolating shared x86-v4 provider COMDAT definitions"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    set_source_files_properties(
        "${_local}"
        PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE
    )
    add_custom_target(${target} DEPENDS "${_local}")
    set(${output} "${_local}" PARENT_SCOPE)
endfunction()
