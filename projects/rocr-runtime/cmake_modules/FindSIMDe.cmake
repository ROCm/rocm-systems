# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include(FindPackageHandleStandardArgs)

find_package(PkgConfig REQUIRED)
if(PkgConfig_FOUND)
    pkg_check_modules(simde IMPORTED_TARGET simde)
endif()

if(PkgConfig_FOUND AND simde_FOUND)
    message(STATUS "Found SIMDe via pkg-config")
    set(SIMDE_TARGET PkgConfig::SIMDE)
else()
    message(STATUS "SIMDe not found via pkg-config. Falling back to find_path...")

    if(WIN32)
        find_path(SIMDE_INCLUDE_DIR
            NAMES simde/simde-common.h
            PATHS
                "C:/simde"
            ENV INCLUDE
    )
    elseif(UNIX)
        find_path(SIMDE_INCLUDE_DIR
            NAMES simde/simde-common.h
            PATHS
                /usr/include
                /usr/local/include
            NO_DEFAULT_PATH
    )
    endif()

    find_package_handle_standard_args(SIMDe
                   REQUIRED_VARS SIMDE_INCLUDE_DIR)
    if(SIMDE_FOUND)
        message(STATUS "Found SIMDe headers at: ${SIMDE_INCLUDE_DIR}")
        if(NOT TARGET SIMDE)
            add_library(SIMDE INTERFACE)
            target_include_directories(SIMDE INTERFACE ${SIMDE_INCLUDE_DIR})
        endif()
    else()
        message(WARNING "could not find simde")
    endif()
endif()