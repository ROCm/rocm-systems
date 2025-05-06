# Generate rust-wrapper and package targets

# This string is the installation directory containing all rust files
set(RUST_WRAPPER_INSTALL_DIR "${SHARE_INSTALL_PREFIX}/rust-wrapper" CACHE STRING "Rust wrapper installation directory")

# Optional to build examples
option(BUILD_RUST_EXAMPLES "Build rust-interface examples" OFF)

# Optional to regenerate the Rust wrapper amdsmi_wrapper.rs
option(REGENERATE_RUST_WRAPPER "Re-generate AMD SMI Rust wrapper file" OFF)

# Determine the Cargo build type based on the CMake build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE debug)
    set(CARGO_BUILD_TYPE_ARG) # Default is debug for Cargo, no argument "--debug"
else()
    set(CARGO_BUILD_TYPE release)
    set(CARGO_BUILD_TYPE_ARG --release)
endif()

set(RUST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(RUST_PACKAGE_SROUCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/source)
set(RUST_OUTPUT_DIR ${RUST_BUILD_DIR}/${CARGO_BUILD_TYPE})

if(REGENERATE_RUST_WRAPPER)
    set(AMDSMI_GENERATE_RUST_WRAPPER "AMDSMI_GENERATE_RUST_WRAPPER=1")
else()
    set(AMDSMI_GENERATE_RUST_WRAPPER "")
endif()

# Create source tar.gz archive
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/source.tar.gz
    COMMAND ${CMAKE_COMMAND} -E make_directory ${RUST_PACKAGE_SROUCE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${RUST_PACKAGE_SROUCE_DIR}/include/amd_smi
    COMMAND ${CMAKE_COMMAND} -E make_directory ${RUST_PACKAGE_SROUCE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
            ${RUST_PACKAGE_SROUCE_DIR}/include/amd_smi/
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/src/libamd_smi.so ${RUST_PACKAGE_SROUCE_DIR}/lib/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/build.rs ${RUST_PACKAGE_SROUCE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/callbacks.rs ${RUST_PACKAGE_SROUCE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml ${RUST_PACKAGE_SROUCE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${RUST_PACKAGE_SROUCE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${RUST_PACKAGE_SROUCE_DIR}/src
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/examples ${RUST_PACKAGE_SROUCE_DIR}/examples
    COMMAND ${CMAKE_COMMAND} -E tar cfz ${CMAKE_CURRENT_BINARY_DIR}/source.tar.gz --format=gnutar
            ${RUST_PACKAGE_SROUCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${AMD_SMI}
            ${CMAKE_CURRENT_SOURCE_DIR}/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/callbacks.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
            ${CMAKE_CURRENT_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/examples
    COMMENT "Creating AMD SMI Rust wrapper source tar.gz archive")

add_custom_target(amdsmi_rust_source ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/source.tar.gz)

find_program(CARGO_EXECUTABLE NAMES cargo)

if(NOT CARGO_EXECUTABLE)
    message(STATUS "Cargo not found. Installing Rust and Cargo...")
    execute_process(COMMAND /bin/sh -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
                    RESULT_VARIABLE result)

    if(result)
        message(FATAL_ERROR "Failed to install Rust and Cargo.")
    endif()

    # Add Cargo to the PATH
    set(CARGO_EXECUTABLE "$ENV{HOME}/.cargo/bin/cargo")
endif()

# Determine the Cargo build command
set(CARGO_BUILD_COMMAND ${CARGO_EXECUTABLE} build ${CARGO_BUILD_TYPE_ARG} --target-dir ${RUST_BUILD_DIR})

# Use make amdsmi_rust to build the library
add_custom_target(
    amdsmi_rust
    COMMAND ${CMAKE_COMMAND} -E env ${AMDSMI_GENERATE_RUST_WRAPPER} AMDSMI_LIB_DIR=${PROJECT_BINARY_DIR}/src
            ${CARGO_BUILD_COMMAND}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${AMD_SMI}
    COMMENT "Build the AMD SMI Rust library")

if(BUILD_RUST_EXAMPLES)
    # Determine the build command for examples if enabled
    set(CARGO_BUILD_EXAMPLES_COMMAND ${CARGO_EXECUTABLE} build ${CARGO_BUILD_TYPE_ARG} --target-dir ${RUST_BUILD_DIR}
                                     --examples)

    # Define the build command for the Rust examples
    add_custom_command(
        OUTPUT ${RUST_OUTPUT_DIR}/examples/amdsmi_get_gpu_info ${RUST_OUTPUT_DIR}/examples/amdsmi_exporter
        COMMAND ${CMAKE_COMMAND} -E env AMDSMI_LIB_DIR=${PROJECT_BINARY_DIR}/src ${CARGO_BUILD_COMMAND}
        COMMAND ${CMAKE_COMMAND} -E env AMDSMI_LIB_DIR=${PROJECT_BINARY_DIR}/src ${CARGO_BUILD_EXAMPLES_COMMAND}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building AMD SMI Rust examples")
    add_custom_target(rust_examples ALL DEPENDS ${RUST_BUILD_DIR}/${CARGO_BUILD_TYPE}/examples/amdsmi_get_gpu_info
                                                ${RUST_OUTPUT_DIR}/examples/amdsmi_exporter)
endif()

# Deploy the Rust interface with source code only, as using static or dynamic library binaries is not recommended by the Rust ecosystem.
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/source.tar.gz
    DESTINATION ${RUST_WRAPPER_INSTALL_DIR}
    COMPONENT dev)

# Deploy rust-interface examples
if(BUILD_RUST_EXAMPLES)
    install(DIRECTORY DESTINATION ${RUST_WRAPPER_INSTALL_DIR}/examples COMPONENT dev)
    install(
        FILES ${RUST_OUTPUT_DIR}/examples/amdsmi_get_gpu_info ${RUST_OUTPUT_DIR}/examples/amdsmi_exporter
        DESTINATION ${RUST_WRAPPER_INSTALL_DIR}/examples
        PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
        COMPONENT dev)
endif()
