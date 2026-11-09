# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Cargo owns Rust dependency tracking; CMake owns native consumers and staging.
include_guard(GLOBAL)
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

macro(runtime_rust_initialize)
    if(WIN32)
        message(
            FATAL_ERROR
            "Rust runtime Windows DLL/import-library staging is not implemented yet"
        )
    elseif(APPLE)
        message(
            FATAL_ERROR
            "Rust runtime Darwin install-name staging is not implemented yet"
        )
    elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(
            FATAL_ERROR
            "Rust runtime staging is not implemented for ${CMAKE_SYSTEM_NAME}"
        )
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(
            FATAL_ERROR
            "Rust runtime cross-compilation is not implemented yet"
        )
    endif()
    set(_runtime_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_runtime_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")
    set(_runtime_library_dir "${_runtime_binary_dir}/lib")
    find_program(ROCM_RUNTIMES_CARGO NAMES cargo REQUIRED)
    find_program(ROCM_RUNTIMES_RUSTC NAMES rustc REQUIRED)
    find_program(ROCM_RUNTIMES_LLD NAMES ld.lld REQUIRED)
    set(ROCM_RUNTIMES_CARGO_HOME
        "${_runtime_binary_dir}/cargo-home"
        CACHE PATH
        "Writable Cargo home (dependency sources must be prepared before configuring)"
    )
    set(ROCM_RUNTIMES_CARGO_CONFIG
        ""
        CACHE FILEPATH
        "Optional prepared Cargo configuration"
    )
    set(ROCM_RUNTIMES_CARGO_JOBS
        ""
        CACHE STRING
        "Optional Cargo parallel-job limit"
    )
    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}"
            "${_runtime_source_dir}/cmake/configure_rust.py" --source
            "${_runtime_source_dir}" --binary "${_runtime_binary_dir}" --cargo
            "${ROCM_RUNTIMES_CARGO}" --rustc "${ROCM_RUNTIMES_RUSTC}" --lld
            "${ROCM_RUNTIMES_LLD}" --cmake "${CMAKE_COMMAND}" --compiler
            "${CMAKE_C_COMPILER}" --processor "${CMAKE_SYSTEM_PROCESSOR}"
            --system "${CMAKE_SYSTEM_NAME}" --build-type "${CMAKE_BUILD_TYPE}"
            --cargo-home "${ROCM_RUNTIMES_CARGO_HOME}" --cargo-config
            "${ROCM_RUNTIMES_CARGO_CONFIG}" --jobs "${ROCM_RUNTIMES_CARGO_JOBS}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    include("${_runtime_binary_dir}/runtime-rust-config.cmake")
    # Regeneration may run from a different directory than initial configure.
    set(ROCM_RUNTIMES_CARGO_CONFIG
        "${_runtime_cargo_config}"
        CACHE FILEPATH
        "Optional prepared Cargo configuration"
        FORCE
    )
    file(
        GLOB_RECURSE _runtime_manifests
        CONFIGURE_DEPENDS
        "${_runtime_source_dir}/Cargo.toml"
    )
    set_property(
        DIRECTORY
        APPEND
        PROPERTY
            CMAKE_CONFIGURE_DEPENDS
                ${_runtime_manifests}
                "${_runtime_source_dir}/Cargo.lock"
                "${_runtime_source_dir}/rust-toolchain.toml"
                "${_runtime_source_dir}/cmake/configure_rust.py"
    )
    if(ROCM_RUNTIMES_CARGO_CONFIG)
        set_property(
            DIRECTORY
            APPEND
            PROPERTY CMAKE_CONFIGURE_DEPENDS "${ROCM_RUNTIMES_CARGO_CONFIG}"
        )
    endif()
    file(MAKE_DIRECTORY "${_runtime_library_dir}")
endmacro()

# VERSION/SOVERSION describe the C ABI, not the Cargo package version.
# SONAME_ENV is consumed by the crate's build.rs.
function(runtime_rust_library target)
    cmake_parse_arguments(
        PARSE_ARGV
        1
        ARG
        ""
        "PACKAGE;LIBRARY;TYPE;OUTPUT_NAME;VERSION;SOVERSION;SONAME_ENV"
        ""
    )
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(
            FATAL_ERROR
            "Invalid runtime_rust_library arguments for ${target}"
        )
    endif()
    foreach(required PACKAGE LIBRARY TYPE OUTPUT_NAME)
        if(NOT DEFINED ARG_${required})
            message(
                FATAL_ERROR
                "runtime_rust_library(${target}) requires ${required}"
            )
        endif()
    endforeach()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(
            FATAL_ERROR
            "runtime_rust_library naming/linking is not implemented for ${CMAKE_SYSTEM_NAME}"
        )
    endif()
    if(ARG_TYPE STREQUAL "SHARED")
        foreach(required VERSION SOVERSION SONAME_ENV)
            if(NOT DEFINED ARG_${required})
                message(
                    FATAL_ERROR
                    "Shared runtime ${target} requires ${required}"
                )
            endif()
        endforeach()
        set(_suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(_link_name
            "${CMAKE_SHARED_LIBRARY_PREFIX}${ARG_OUTPUT_NAME}${_suffix}"
        )
        set(_soname "${_link_name}.${ARG_SOVERSION}")
        set(_filename "${_link_name}.${ARG_VERSION}")
        if(NOT _filename STREQUAL _soname)
            set_property(
                GLOBAL
                APPEND
                PROPERTY
                    _runtime_link_commands
                        COMMAND
                        "${CMAKE_COMMAND}"
                        -E
                        create_symlink
                        "${_filename}"
                        "${_runtime_library_dir}/${_soname}"
            )
            set_property(
                GLOBAL
                APPEND
                PROPERTY
                    _runtime_byproducts "${_runtime_library_dir}/${_soname}"
            )
        endif()
        set_property(
            GLOBAL
            APPEND
            PROPERTY
                _runtime_link_commands
                    COMMAND
                    "${CMAKE_COMMAND}"
                    -E
                    create_symlink
                    "${_soname}"
                    "${_runtime_library_dir}/${_link_name}"
        )
        set_property(
            GLOBAL
            APPEND
            PROPERTY _runtime_byproducts "${_runtime_library_dir}/${_link_name}"
        )
        set_property(
            GLOBAL
            APPEND
            PROPERTY _runtime_soname_env "${ARG_SONAME_ENV}=${_soname}"
        )
    elseif(ARG_TYPE STREQUAL "STATIC")
        set(_suffix "${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(_filename
            "${CMAKE_STATIC_LIBRARY_PREFIX}${ARG_OUTPUT_NAME}${_suffix}"
        )
    else()
        message(
            FATAL_ERROR
            "runtime_rust_library TYPE must be SHARED or STATIC"
        )
    endif()
    add_library(${target} ${ARG_TYPE} IMPORTED GLOBAL)
    set_target_properties(
        ${target}
        PROPERTIES IMPORTED_LOCATION "${_runtime_library_dir}/${_filename}"
    )
    add_dependencies(${target} rocm_runtime_rust)
    if(ARG_TYPE STREQUAL "SHARED")
        set_target_properties(${target} PROPERTIES IMPORTED_SONAME "${_soname}")
    else()
        set(_response "${_runtime_library_dir}/${target}-native-libs.rsp")
        set_property(
            GLOBAL
            APPEND
            PROPERTY
                _runtime_native_args
                    --native-libs
                    "${ARG_PACKAGE}"
                    "${ARG_LIBRARY}"
                    "${_response}"
        )
        set_property(GLOBAL APPEND PROPERTY _runtime_byproducts "${_response}")
        # GNU response-file contents must follow the archive, not precede it as options.
        target_link_libraries(${target} INTERFACE "-Wl,@${_response}")
        set_property(
            TARGET ${target}
            APPEND
            PROPERTY INTERFACE_LINK_DEPENDS "${_response}"
        )
    endif()
    set_property(
        GLOBAL
        APPEND
        PROPERTY
            _runtime_artifact_args
                --artifact
                "${ARG_PACKAGE}"
                "${ARG_LIBRARY}"
                "${_suffix}"
                "${_runtime_library_dir}/${_filename}"
    )
    set_property(
        GLOBAL
        APPEND
        PROPERTY _runtime_byproducts "${_runtime_library_dir}/${_filename}"
    )
endfunction()

function(runtime_rust_finalize)
    get_property(_artifacts GLOBAL PROPERTY _runtime_artifact_args)
    get_property(_native_args GLOBAL PROPERTY _runtime_native_args)
    get_property(_byproducts GLOBAL PROPERTY _runtime_byproducts)
    get_property(_links GLOBAL PROPERTY _runtime_link_commands)
    get_property(_soname_env GLOBAL PROPERTY _runtime_soname_env)
    set(_command
        "${CMAKE_COMMAND}"
        -E
        env
        ${_soname_env}
        ${_runtime_cargo_command}
    )
    set(_runtime_cargo_command ${_command} PARENT_SCOPE)
    add_custom_target(
        rocm_runtime_rust
        ALL
        COMMAND
            "${Python3_EXECUTABLE}"
            "${_runtime_source_dir}/cmake/cargo_artifacts.py" --metadata
            "${_runtime_binary_dir}/cargo-metadata.json" ${_artifacts}
            ${_native_args} -- ${_command} build --workspace --frozen --profile
            "${_runtime_profile}" --message-format=json ${_links}
        BYPRODUCTS ${_byproducts}
        COMMENT "Building runtime Rust libraries and staging native artifacts"
        WORKING_DIRECTORY "${_runtime_source_dir}"
        VERBATIM
        USES_TERMINAL
    )
endfunction()
