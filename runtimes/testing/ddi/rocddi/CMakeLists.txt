# SPDX-License-Identifier: MIT

find_program(ROCM_RUNTIMES_CARGO cargo REQUIRED)
find_program(ROCM_RUNTIMES_RUSTC rustc REQUIRED)
get_filename_component(
    _workspace
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
    ABSOLUTE
)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_workspace}/rust-toolchain.toml"
)
file(STRINGS "${_workspace}/rust-toolchain.toml" _channel REGEX "^channel =")
string(REGEX REPLACE "^channel = \"([^\"]+)\"$" "\\1" _version "${_channel}")
set(_cargo_environment
    "RUSTUP_TOOLCHAIN=${_version}"
    RUSTUP_AUTO_INSTALL=0
    CARGO_NET_OFFLINE=true
    "RUSTC=${ROCM_RUNTIMES_RUSTC}"
    "CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}/cargo"
)
if(DEFINED ROCM_RUNTIMES_CARGO_HOME)
    if(
        NOT IS_ABSOLUTE "${ROCM_RUNTIMES_CARGO_HOME}"
        OR NOT IS_DIRECTORY "${ROCM_RUNTIMES_CARGO_HOME}"
    )
        message(
            FATAL_ERROR
            "ROCM_RUNTIMES_CARGO_HOME must be an absolute path to a prepared directory"
        )
    endif()
    list(APPEND _cargo_environment "CARGO_HOME=${ROCM_RUNTIMES_CARGO_HOME}")
endif()
set(_cargo_command
    "${CMAKE_COMMAND}"
    -E
    env
    ${_cargo_environment}
    "${ROCM_RUNTIMES_CARGO}"
)
set(_config "")
if(
    DEFINED ROCM_RUNTIMES_CARGO_CONFIG
    AND NOT ROCM_RUNTIMES_CARGO_CONFIG STREQUAL ""
)
    set(_config "${ROCM_RUNTIMES_CARGO_CONFIG}")
    if(
        NOT IS_ABSOLUTE "${_config}"
        OR NOT EXISTS "${_config}"
        OR IS_DIRECTORY "${_config}"
    )
        message(
            FATAL_ERROR
            "ROCM_RUNTIMES_CARGO_CONFIG must be an absolute path to a prepared file"
        )
    endif()
    list(APPEND _cargo_command --config "${_config}")
endif()
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env ${_cargo_environment} "${ROCM_RUNTIMES_RUSTC}"
        --version --verbose
    WORKING_DIRECTORY "${_workspace}"
    OUTPUT_VARIABLE _rust_info
    COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX MATCH "release: ([^\n]+)" _release_match "${_rust_info}")
if(NOT CMAKE_MATCH_1 STREQUAL _version)
    message(
        FATAL_ERROR
        "DDI tests require prepared Rust ${_version}; got ${_rust_info}"
    )
endif()
execute_process(
    COMMAND ${_cargo_command} --version
    WORKING_DIRECTORY "${_workspace}"
    OUTPUT_VARIABLE _cargo_info
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX MATCH "^cargo ([^ ]+) " _cargo_match "${_cargo_info}")
if(NOT CMAKE_MATCH_1 STREQUAL _version)
    message(
        FATAL_ERROR
        "DDI tests require prepared Cargo ${_version}; got ${_cargo_info}"
    )
endif()
string(REGEX MATCH "host: ([^\n]+)" _match "${_rust_info}")
set(_host "${CMAKE_MATCH_1}")
if(
    NOT (
        CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"
        AND _host STREQUAL "x86_64-unknown-linux-gnu"
    )
    AND NOT (
        CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"
        AND _host STREQUAL "aarch64-unknown-linux-gnu"
    )
)
    message(
        FATAL_ERROR
        "Unsupported or mismatched native DDI target: ${CMAKE_SYSTEM_PROCESSOR} / ${_host}"
    )
endif()
set(_output "${runtime_test_prefix}/bin/ddi/rocddi-host-contract")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_profile dev)
    set(_profile_directory debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_profile release)
    set(_profile_directory release)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(_profile relwithdebinfo)
    set(_profile_directory relwithdebinfo)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(_profile minsizerel)
    set(_profile_directory minsizerel)
else()
    message(FATAL_ERROR "Unsupported DDI build type: ${CMAKE_BUILD_TYPE}")
endif()
add_custom_target(
    rocddi-host-contract
    ALL
    COMMAND
        ${_cargo_command} build --frozen --offline --package
        rocddi-contract-tests --bin rocddi-host-contract --profile "${_profile}"
        --target "${_host}"
    COMMAND
        "${CMAKE_COMMAND}" -E make_directory "${runtime_test_prefix}/bin/ddi"
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/cargo/${_host}/${_profile_directory}/rocddi-host-contract"
        "${_output}"
    COMMAND
        "${CMAKE_COMMAND}" "-DTEST_EXECUTABLE=${_output}"
        "-DRECORD=${runtime_test_prefix}/share/rocm-runtime-tests/ddi.txt"
        "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DEXTRA_INPUT_ROOTS=${_workspace}/ddi/rocddi/src;${_workspace}/ddi/rocddi/Cargo.toml;${_workspace}/Cargo.toml;${_workspace}/Cargo.lock;${_workspace}/rust-toolchain.toml;${PROJECT_SOURCE_DIR}/cmake;${PROJECT_SOURCE_DIR}/CMakeLists.txt;${_config}"
        "-DCOMPILER=rustc ${_version}; profile=${_profile}; target=${_host}" -P
        "${PROJECT_SOURCE_DIR}/cmake/RecordBuild.cmake"
    BYPRODUCTS "${_output}"
    WORKING_DIRECTORY "${_workspace}"
    VERBATIM
)
runtime_test_register_suite(ddi-host ddi rocddi-host-contract ddi.rocddi.host-allocation FALSE)
