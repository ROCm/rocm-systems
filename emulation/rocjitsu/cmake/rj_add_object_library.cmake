# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define an OBJECT library with standard include dirs and flags for
# rocjitsu sub-components. ROCJITSU_INCLUDE_DIR and ROCJITSU_SRC_DIR
# must be set before including this module.
#
# Apply the common include paths and warnings to one rocjitsu object library.
function(_rj_configure_object_library name)
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
        ${name}
        PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
    )
    target_link_libraries(${name} PRIVATE rocm_runtime::runtime_headers ${ARGN})
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(
            ${name}
            PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden
        )
        if(
            CMAKE_CXX_COMPILER_ID MATCHES "GNU"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15
        )
            target_compile_options(
                ${name}
                PRIVATE -Wno-error=maybe-uninitialized
            )
        endif()
    endif()
endfunction()

# Usage: rj_add_object_library(<name> <sources...>)
function(rj_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    _rj_configure_object_library(${name} util simdojo_headers)
endfunction()

# Select objects for a shared object that does not link the simulator image.
# UBSan's vptr check references the RTTI of each class it checks, and a class
# with a key function emits its RTTI only in the source that defines that
# function. Such a shared object can therefore have undefined RTTI references.
# When the check is enabled, compile a copy of <object-lib> without it and
# return the copy; otherwise return <object-lib>. The original objects keep the
# check wherever the simulator is linked, including rocjitsu_tests.
#
# The copy has the original's sources and private link libraries. Interface
# link libraries are not copied, so callers must add any objects that the
# original exposes through its interface.
#
# Usage: rj_vptr_free_object_library(<out-var> <object-lib>)
function(rj_vptr_free_object_library out_var object_lib)
    if(NOT (RJ_ENABLE_UBSAN AND RJ_UBSAN_VPTR))
        set(${out_var} ${object_lib} PARENT_SCOPE)
        return()
    endif()
    set(_copy ${object_lib}_no_vptr)
    if(NOT TARGET ${_copy})
        get_target_property(_source_dir ${object_lib} SOURCE_DIR)
        get_target_property(_sources ${object_lib} SOURCES)
        set(_absolute_sources)
        foreach(_source IN LISTS _sources)
            cmake_path(
                ABSOLUTE_PATH _source
                BASE_DIRECTORY "${_source_dir}"
                NORMALIZE
            )
            list(APPEND _absolute_sources "${_source}")
        endforeach()
        rj_add_object_library(${_copy} ${_absolute_sources})
        target_link_libraries(
            ${_copy}
            PRIVATE $<TARGET_PROPERTY:${object_lib},LINK_LIBRARIES>
        )
        target_compile_options(${_copy} PRIVATE -fno-sanitize=vptr)
    endif()
    set(${out_var} ${_copy} PARENT_SCOPE)
endfunction()

# Mark an existing ISA implementation target as a statically linked provider.
# The target owns its provider source and self-contained declaration header;
# CMake only records the header used to compose consumer-specific registries.
#
# Usage:
#   rj_add_isa_target_provider(
#       <implementation-target>
#       HEADER <provider-header>
#       [BUILTIN]
#   )
function(rj_add_isa_target_provider name)
    set(options BUILTIN)
    set(oneValueArgs HEADER)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
    if(NOT ARG_HEADER)
        message(
            FATAL_ERROR
            "rj_add_isa_target_provider(${name}) requires HEADER"
        )
    endif()
    if(NOT TARGET ${name})
        message(FATAL_ERROR "unknown ISA implementation target '${name}'")
    endif()
    get_target_property(_provider_header ${name} RJ_ISA_PROVIDER_HEADER)
    if(_provider_header)
        message(FATAL_ERROR "target '${name}' is already an ISA provider")
    endif()
    set_target_properties(
        ${name}
        PROPERTIES RJ_ISA_PROVIDER_HEADER "${ARG_HEADER}"
    )
    if(ARG_BUILTIN)
        set_property(GLOBAL APPEND PROPERTY RJ_BUILTIN_ISA_PROVIDERS ${name})
    endif()
endfunction()

# Assemble one final image's immutable default registry from an exact provider
# list. Provider order is preserved in the generated include list and therefore
# in registry enumeration. Checked-in C++ owns registry construction and public
# enum lookup.
function(rj_add_isa_target_registry name)
    cmake_parse_arguments(ARG "" "" "PROVIDERS" ${ARGN})
    if(NOT ARG_PROVIDERS)
        message(
            FATAL_ERROR
            "rj_add_isa_target_registry(${name}) requires PROVIDERS"
        )
    endif()

    set(_target_headers_content)
    set(_provider_objects)
    # Iterating the caller's list directly preserves collection order. Each
    # included provider contributes its descriptor at that same position.
    foreach(_provider IN LISTS ARG_PROVIDERS)
        if(NOT TARGET ${_provider})
            message(FATAL_ERROR "unknown ISA provider target '${_provider}'")
        endif()
        get_target_property(
            _provider_header
            ${_provider}
            RJ_ISA_PROVIDER_HEADER
        )
        if(NOT _provider_header)
            message(FATAL_ERROR "target '${_provider}' is not an ISA provider")
        endif()
        string(
            APPEND _target_headers_content
            "#include \"${_provider_header}\"\n"
        )
        get_target_property(_provider_type ${_provider} TYPE)
        if(_provider_type STREQUAL "OBJECT_LIBRARY")
            list(APPEND _provider_objects $<TARGET_OBJECTS:${_provider}>)
        endif()
    endforeach()

    set(_target_headers_name "${name}_targets.h")
    set(_target_headers "${CMAKE_CURRENT_BINARY_DIR}/${_target_headers_name}")
    file(
        GENERATE OUTPUT "${_target_headers}"
        CONTENT "${_target_headers_content}"
    )

    set(_composition_target "${name}__composition")
    add_library(
        ${_composition_target}
        OBJECT
        "${ROCJITSU_SRC_DIR}/rocjitsu/isa/target_registry_composition.cpp"
    )
    _rj_configure_object_library(
        ${_composition_target}
        util
        simdojo_headers
        ${ARG_PROVIDERS}
    )
    target_include_directories(
        ${_composition_target}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    )
    target_compile_definitions(
        ${_composition_target}
        PRIVATE "RJ_ISA_TARGET_HEADERS=\"${_target_headers_name}\""
    )
    add_library(${name} INTERFACE)
    target_sources(${name} INTERFACE $<TARGET_OBJECTS:${_composition_target}>)
    if(_provider_objects)
        target_sources(${name} INTERFACE ${_provider_objects})
    endif()
    target_link_libraries(${name} INTERFACE ${ARG_PROVIDERS})
endfunction()
