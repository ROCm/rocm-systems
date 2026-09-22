cmake_minimum_required(VERSION 3.16)
project(perf-pmu-stub VERSION 1.0.0 LANGUAGES C)

# Add cmake modules to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Include GNUInstallDirs for standard installation paths
include(GNUInstallDirs)

#================================================================================
# Build Options
#================================================================================

option(BUILD_KERNEL_MODULE "Build the kernel module component" ON)
option(BUILD_USERSPACE_TOOLS "Build packet_gen_tool and pm4_decoder" ON)
option(BUILD_TESTS "Build test executables" ON)
option(ENABLE_DEBUG "Enable debug flags (-DDEBUG, -g)" ON)
option(ENABLE_DKMS "Enable DKMS installation targets" OFF)

#================================================================================
# Module Configuration
#================================================================================

set(MODULE_NAME "perf-pmu-stub" CACHE STRING "Kernel module name")
set(MODULE_VERSION "${PROJECT_VERSION}" CACHE STRING "Module version")

#================================================================================
# Kernel Configuration
#================================================================================

# Find kernel headers if building kernel module
if(BUILD_KERNEL_MODULE)
    find_package(KernelHeaders REQUIRED)

    # Set DKMS installation directory
    if(NOT DKMS_INSTALL_DIR)
        set(DKMS_INSTALL_DIR "/usr/src" CACHE PATH "DKMS source installation directory")
    endif()
endif()

#================================================================================
# Compiler Configuration
#================================================================================

# Set C standard for userspace code
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Debug flags
if(ENABLE_DEBUG)
    add_compile_options(-g -Wall -Wextra)
    add_compile_definitions(DEBUG)
endif()

#================================================================================
# Build Type
#================================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

#================================================================================
# Subdirectories
#================================================================================

if(BUILD_KERNEL_MODULE)
    add_subdirectory(src)
endif()

if(BUILD_USERSPACE_TOOLS)
    add_subdirectory(src/aql_c/tools)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(src/aql_c/tests)
endif()

#================================================================================
# DKMS Targets
#================================================================================

if(ENABLE_DKMS AND BUILD_KERNEL_MODULE)
    set(DKMS_SOURCE_PATH "${DKMS_INSTALL_DIR}/${MODULE_NAME}-${MODULE_VERSION}")

    add_custom_target(dkms-add
        COMMAND ${CMAKE_COMMAND} -E echo "Installing source to ${DKMS_SOURCE_PATH}"
        COMMAND sudo cp -R ${CMAKE_CURRENT_SOURCE_DIR} ${DKMS_SOURCE_PATH}
        COMMAND sudo dkms add -m ${MODULE_NAME} -v ${MODULE_VERSION}
        COMMENT "Adding module to DKMS"
    )

    add_custom_target(dkms-build
        COMMAND sudo dkms build -m ${MODULE_NAME} -v ${MODULE_VERSION}
        DEPENDS dkms-add
        COMMENT "Building module with DKMS"
    )

    add_custom_target(dkms-install
        COMMAND sudo dkms install -m ${MODULE_NAME} -v ${MODULE_VERSION}
        DEPENDS dkms-build
        COMMENT "Installing module with DKMS"
    )

    add_custom_target(dkms-remove
        COMMAND sudo dkms remove -m ${MODULE_NAME} -v ${MODULE_VERSION} --all || true
        COMMAND sudo rm -rf ${DKMS_SOURCE_PATH} || true
        COMMENT "Removing module from DKMS"
    )
endif()

#================================================================================
# Information Display
#================================================================================

message(STATUS "")
message(STATUS "============================================")
message(STATUS "  perf-pmu-stub Build Configuration")
message(STATUS "============================================")
message(STATUS "Module Name:           ${MODULE_NAME}")
message(STATUS "Module Version:        ${MODULE_VERSION}")
message(STATUS "Build Type:            ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Kernel Module:       ${BUILD_KERNEL_MODULE}")
message(STATUS "  Userspace Tools:     ${BUILD_USERSPACE_TOOLS}")
message(STATUS "  Tests:               ${BUILD_TESTS}")
message(STATUS "  Debug:               ${ENABLE_DEBUG}")
message(STATUS "  DKMS:                ${ENABLE_DKMS}")
if(BUILD_KERNEL_MODULE)
    message(STATUS "")
    message(STATUS "Kernel Configuration:")
    message(STATUS "  Kernel Version:      ${KERNEL_VERSION}")
    message(STATUS "  Kernel Build Dir:    ${KERNEL_BUILD_DIR}")
    if(ENABLE_DKMS)
        message(STATUS "  DKMS Install Dir:    ${DKMS_INSTALL_DIR}")
    endif()
endif()
message(STATUS "============================================")
message(STATUS "")
