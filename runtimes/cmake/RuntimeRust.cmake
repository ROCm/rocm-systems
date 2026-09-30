# Build-tree integration only. Cargo owns Rust dependency tracking; CMake owns
# native consumers. There are deliberately no install/export rules yet.
include_guard(GLOBAL)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

macro(runtime_rust_initialize)
  find_program(ROCM_RUNTIMES_CARGO NAMES cargo REQUIRED)
  find_program(ROCM_RUNTIMES_RUSTC NAMES rustc REQUIRED)
  find_program(ROCM_RUNTIMES_LLD NAMES ld.lld REQUIRED)
  execute_process(COMMAND "${ROCM_RUNTIMES_LLD}" --version
    RESULT_VARIABLE _runtime_result OUTPUT_QUIET ERROR_VARIABLE _runtime_error)
  if(NOT _runtime_result EQUAL 0)
    message(FATAL_ERROR "ROCM_RUNTIMES_LLD must name a working ld.lld: ${_runtime_error}")
  endif()
  set(ROCM_RUNTIMES_CARGO_HOME "${PROJECT_BINARY_DIR}/cargo-home" CACHE PATH
    "Writable Cargo home (dependency sources must be prepared before configuring)")
  set(ROCM_RUNTIMES_CARGO_CONFIG "" CACHE FILEPATH "Optional prepared Cargo configuration")
  set(ROCM_RUNTIMES_CARGO_JOBS "" CACHE STRING "Optional Cargo parallel job limit")
  if(ROCM_RUNTIMES_CARGO_JOBS AND NOT ROCM_RUNTIMES_CARGO_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "ROCM_RUNTIMES_CARGO_JOBS must be a positive integer")
  endif()
  if(ROCM_RUNTIMES_CARGO_CONFIG AND NOT EXISTS "${ROCM_RUNTIMES_CARGO_CONFIG}")
    message(FATAL_ERROR "Cargo configuration does not exist: ${ROCM_RUNTIMES_CARGO_CONFIG}")
  endif()

  file(READ "${PROJECT_SOURCE_DIR}/rust-toolchain.toml" _runtime_toolchain)
  string(REGEX MATCH "channel = \"([^\"]+)\"" _runtime_match "${_runtime_toolchain}")
  set(_runtime_rust_version "${CMAKE_MATCH_1}")
  if(NOT _runtime_rust_version)
    message(FATAL_ERROR "Cannot read the pinned Rust toolchain version")
  endif()
  # Disable rustup's implicit installation when explicit paths are rustup proxies.
  set(_runtime_env "RUSTUP_TOOLCHAIN=${_runtime_rust_version}" "RUSTUP_AUTO_INSTALL=0"
    "RUSTC=${ROCM_RUNTIMES_RUSTC}" "CARGO_HOME=${ROCM_RUNTIMES_CARGO_HOME}"
    "CARGO_TARGET_DIR=${PROJECT_BINARY_DIR}/cargo" "CARGO_NET_OFFLINE=true")
  get_filename_component(_runtime_lld_dir "${ROCM_RUNTIMES_LLD}" DIRECTORY)
  set(_runtime_path "${_runtime_lld_dir}:$ENV{PATH}")
  list(APPEND _runtime_env "PATH=${_runtime_path}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${_runtime_env}
      "${ROCM_RUNTIMES_RUSTC}" --version --verbose
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE _runtime_rust_info ERROR_VARIABLE _runtime_error RESULT_VARIABLE _runtime_result)
  if(NOT _runtime_result EQUAL 0 OR NOT _runtime_rust_info MATCHES "rustc ${_runtime_rust_version} ")
    message(FATAL_ERROR "Provision Rust ${_runtime_rust_version} before configuring, or set ROCM_RUNTIMES_RUSTC.\n${_runtime_rust_info}\n${_runtime_error}")
  endif()
  string(REGEX MATCH "host: ([^\n\r]+)" _runtime_match "${_runtime_rust_info}")
  set(_runtime_host "${CMAKE_MATCH_1}")
  if((CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND NOT _runtime_host STREQUAL "x86_64-unknown-linux-gnu")
      OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND NOT _runtime_host STREQUAL "aarch64-unknown-linux-gnu"))
    message(FATAL_ERROR "Rust host ${_runtime_host} does not match the native CMake target")
  endif()
  # Do not inherit a developer's cross target from ambient Cargo configuration.
  list(APPEND _runtime_env "CARGO_BUILD_TARGET=${_runtime_host}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${_runtime_env}
      "${ROCM_RUNTIMES_CARGO}" --version
    OUTPUT_VARIABLE _runtime_cargo_version RESULT_VARIABLE _runtime_result)
  if(NOT _runtime_result EQUAL 0 OR NOT _runtime_cargo_version MATCHES "cargo ${_runtime_rust_version} ")
    message(FATAL_ERROR "ROCM_RUNTIMES_CARGO must select Cargo ${_runtime_rust_version}")
  endif()

  # Ask rustc for the static library's actual native closure. The artifact helper
  # records it in a linker response file propagated by the imported static target.
  string(ASCII 31 _runtime_separator)
  if(DEFINED ENV{CARGO_ENCODED_RUSTFLAGS})
    set(_runtime_flags "$ENV{CARGO_ENCODED_RUSTFLAGS}")
  else()
    separate_arguments(_runtime_flag_list UNIX_COMMAND "$ENV{RUSTFLAGS}")
    list(JOIN _runtime_flag_list "${_runtime_separator}" _runtime_flags)
  endif()
  if(_runtime_flags)
    string(APPEND _runtime_flags "${_runtime_separator}")
  endif()
  # HSA's symbol aliases require LLD for its Rust test executables as well as
  # its cdylib. Overriding rustc's linker driver otherwise falls back to ld.bfd.
  string(APPEND _runtime_flags "--print=native-static-libs${_runtime_separator}-C${_runtime_separator}linker=${CMAKE_C_COMPILER}${_runtime_separator}-C${_runtime_separator}link-arg=-fuse-ld=lld")
  list(APPEND _runtime_env "CARGO_ENCODED_RUSTFLAGS=${_runtime_flags}")
  if(ROCM_RUNTIMES_CARGO_JOBS)
    list(APPEND _runtime_env "CARGO_BUILD_JOBS=${ROCM_RUNTIMES_CARGO_JOBS}")
  endif()
  set(_runtime_cargo_command "${CMAKE_COMMAND}" -E env ${_runtime_env} "${ROCM_RUNTIMES_CARGO}")
  if(ROCM_RUNTIMES_CARGO_CONFIG)
    list(APPEND _runtime_cargo_command --config "${ROCM_RUNTIMES_CARGO_CONFIG}")
  endif()
  execute_process(COMMAND ${_runtime_cargo_command} metadata --format-version 1 --no-deps --frozen
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE _runtime_metadata ERROR_VARIABLE _runtime_error RESULT_VARIABLE _runtime_result)
  if(NOT _runtime_result EQUAL 0)
    message(FATAL_ERROR "Frozen Cargo metadata failed; prepare the toolchain/dependencies first.\n${_runtime_error}")
  endif()
  file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/cargo-metadata.json" CONTENT "${_runtime_metadata}" @ONLY)
  file(GLOB_RECURSE _runtime_manifests CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/Cargo.toml")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_runtime_manifests} "${PROJECT_SOURCE_DIR}/Cargo.lock" "${PROJECT_SOURCE_DIR}/rust-toolchain.toml")
  if(ROCM_RUNTIMES_CARGO_CONFIG)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ROCM_RUNTIMES_CARGO_CONFIG}")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_runtime_profile dev)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_runtime_profile release)
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(_runtime_profile relwithdebinfo)
  elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(_runtime_profile minsizerel)
  else()
    message(FATAL_ERROR "Unsupported runtime configuration: ${CMAKE_BUILD_TYPE}")
  endif()
  set(_runtime_library_dir "${PROJECT_BINARY_DIR}/lib")
  file(MAKE_DIRECTORY "${_runtime_library_dir}")
endmacro()

function(runtime_rust_library target package library kind filename)
  add_library(${target} ${kind} IMPORTED GLOBAL)
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION "${_runtime_library_dir}/${filename}")
  add_dependencies(${target} runtime_rust)
  if(kind STREQUAL "STATIC")
    set(_suffix .a)
  else()
    set(_suffix .so)
  endif()
  set_property(GLOBAL APPEND PROPERTY _runtime_artifact_args
    --artifact "${package}" "${library}" "${_suffix}" "${_runtime_library_dir}/${filename}")
  set_property(GLOBAL APPEND PROPERTY _runtime_byproducts "${_runtime_library_dir}/${filename}")
endfunction()

function(runtime_rust_finalize)
  get_property(_artifacts GLOBAL PROPERTY _runtime_artifact_args)
  get_property(_byproducts GLOBAL PROPERTY _runtime_byproducts)
  get_target_property(_hsa_location runtime_hsa_shared IMPORTED_LOCATION)
  get_filename_component(_hsa_filename "${_hsa_location}" NAME)
  get_target_property(_amdf_location runtime_amdf_shared IMPORTED_LOCATION)
  get_filename_component(_amdf_filename "${_amdf_location}" NAME)
  add_custom_target(runtime_rust ALL
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/cargo_artifacts.py"
      --metadata "${PROJECT_BINARY_DIR}/cargo-metadata.json" ${_artifacts}
      --native-libs "${_runtime_library_dir}/amdf-native-libs.rsp"
      -- ${_runtime_cargo_command} build --workspace --frozen
        --profile "${_runtime_profile}" --message-format=json
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_amdf_filename}"
      "${_runtime_library_dir}/libamdf.so.0"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink libamdf.so.0
      "${_runtime_library_dir}/libamdf.so"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_hsa_filename}"
      "${_runtime_library_dir}/libhsa-runtime64.so.1"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink libhsa-runtime64.so.1
      "${_runtime_library_dir}/libhsa-runtime64.so"
    BYPRODUCTS ${_byproducts} "${_runtime_library_dir}/amdf-native-libs.rsp"
      "${_runtime_library_dir}/libamdf.so.0" "${_runtime_library_dir}/libamdf.so"
      "${_runtime_library_dir}/libhsa-runtime64.so.1" "${_runtime_library_dir}/libhsa-runtime64.so"
    COMMENT "Building runtime Rust libraries and staging native artifacts"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" VERBATIM USES_TERMINAL)
endfunction()
