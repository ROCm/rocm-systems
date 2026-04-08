# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# =====================================================================================
# DyninstTBB.cmake
#
# Configure Threading Building Blocks (TBB) for Dyninst
#
# Common CMake inputs: ROCPROFSYS_BUILD_TBB, TBB_ROOT_DIR, TBB_INCLUDEDIR,
# TBB_LIBRARYDIR, TBB_USE_DEBUG_BUILD, TBB_MIN_VERSION.
# Exported cache vars include TBB_INCLUDE_DIRS, TBB_LIBRARY_DIRS, TBB_LIBRARIES.
# Full input/output variable lists and behavior: cmake/Modules/FindTBB.cmake (module docs).
#
# =====================================================================================

include_guard(GLOBAL)

if(TBB_FOUND)
    rocprofiler_systems_message(STATUS "TBB already found, skipping build")
    return()
endif()

# -------------- RUNTIME CONFIGURATION ----------------------------------------

# Use debug versions of TBB libraries
set(TBB_USE_DEBUG_BUILD OFF CACHE BOOL "Use debug versions of TBB libraries")

# Minimum version of TBB (assumes a dotted-decimal format: YYYY.XX)
set(_tbb_min_version 2018.6)

set(TBB_MIN_VERSION
    ${_tbb_min_version}
    CACHE STRING
    "Minimum version of TBB (assumes a dotted-decimal format: YYYY.XX)"
)

if(${TBB_MIN_VERSION} VERSION_LESS ${_tbb_min_version})
    rocprofiler_systems_message(
        FATAL_ERROR
        "Requested TBB version ${TBB_MIN_VERSION} is less than minimum supported version ${_tbb_min_version}"
    )
endif()

# -------------- PATHS --------------------------------------------------------

# TBB root directory
set(TBB_ROOT_DIR "/usr" CACHE PATH "TBB root directory")

# TBB include directory hint
set(TBB_INCLUDEDIR "${TBB_ROOT_DIR}/include" CACHE INTERNAL "TBB include directory")

# TBB library directory hint
set(TBB_LIBRARYDIR "${TBB_ROOT_DIR}/lib" CACHE INTERNAL "TBB library directory")

# Translate to FindTBB names
set(TBB_LIBRARY ${TBB_LIBRARYDIR})
set(TBB_INCLUDE_DIR ${TBB_INCLUDEDIR})

# The specific TBB libraries we need NB: This should _NOT_ be a cache variable
set(_tbb_components tbb tbbmalloc_proxy tbbmalloc)

if(NOT ROCPROFSYS_BUILD_TBB)
    find_package(TBB ${TBB_MIN_VERSION} COMPONENTS ${_tbb_components})
endif()

# -------------- SOURCE BUILD -------------------------------------------------
if(TBB_FOUND)
    # Force the cache entries to be updated Normally, these would not be exported.
    # However, we need them in the Testsuite
    set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIRS} CACHE PATH "TBB include directory" FORCE)
    set(TBB_LIBRARY_DIRS ${TBB_LIBRARY_DIRS} CACHE PATH "TBB library directory" FORCE)
    set(TBB_DEFINITIONS ${TBB_DEFINITIONS} CACHE STRING "TBB compiler definitions" FORCE)
    set(TBB_LIBRARIES ${TBB_LIBRARIES} CACHE FILEPATH "TBB library files" FORCE)

    # Update TBB_ROOT_DIR to the found location for Dyninst.
    # Prefer include dirs: multiarch TBB_DIR under lib/<triplet>/cmake/... breaks fixed depth.
    set(_tbb_root "")
    if(TBB_INCLUDE_DIRS)
        list(GET TBB_INCLUDE_DIRS 0 _tbb_inc)
        get_filename_component(_tbb_root "${_tbb_inc}" DIRECTORY)
    endif()
    if(NOT _tbb_root AND TBB_DIR)
        string(REGEX REPLACE "/lib(/[^/]+)*/cmake/TBB[^/]*$" "" _tbb_root "${TBB_DIR}")
        if(_tbb_root STREQUAL TBB_DIR)
            set(_tbb_root "")
        else()
            get_filename_component(_tbb_root "${_tbb_root}" ABSOLUTE)
        endif()
    endif()
    if(_tbb_root)
        set(TBB_ROOT_DIR "${_tbb_root}" CACHE PATH "TBB root directory" FORCE)
    endif()
    set(TBB_ROOT ${TBB_ROOT_DIR})
elseif(STERILE_BUILD)
    rocprofiler_systems_message(
        FATAL_ERROR
            "TBB not found and cannot be built from the external submodule because the build is sterile."
    )
elseif(NOT ROCPROFSYS_BUILD_TBB)
    rocprofiler_systems_message(
        FATAL_ERROR
            "TBB was not found. Either configure CMake to find a suitable system TBB or set ROCPROFSYS_BUILD_TBB=ON to build from the external/onetbb submodule"
    )
else()
    # Verify oneTBB submodule is initialized
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/onetbb
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/uxlfoundation/oneTBB.git
        REPO_BRANCH "v2022.3.0"
    )

    # If we didn't find a suitable version on the system, then build from submodule
    rocprofiler_systems_message(STATUS "${ThreadingBuildingBlocks_ERROR_REASON}")
    rocprofiler_systems_message(
        STATUS "Attempting to build TBB(${TBB_MIN_VERSION}) from external/onetbb submodule"
    )

    if(NOT UNIX)
        rocprofiler_systems_message(
            FATAL_ERROR "Building TBB from source is not supported on this platform"
        )
    endif()

    set(TBB_ROOT_DIR ${TPL_STAGING_PREFIX}/tbb CACHE PATH "TBB root directory" FORCE)

    set(_tbb_libraries)
    set(_tbb_build_byproducts)
    set(_tbb_library_dirs
        $<BUILD_INTERFACE:${TBB_ROOT_DIR}/lib>
        $<INSTALL_INTERFACE:${INSTALL_LIB_DIR}/${TPL_INSTALL_LIB_DIR}>
    )
    set(_tbb_include_dirs
        $<BUILD_INTERFACE:${TBB_ROOT_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_LIB_DIR}/${TPL_INSTALL_INCLUDE_DIR}>
    )

    # Forcibly update the cache variables
    set(TBB_INCLUDE_DIRS "${_tbb_include_dirs}" CACHE PATH "TBB include directory" FORCE)
    set(TBB_LIBRARY_DIRS "${_tbb_library_dirs}" CACHE PATH "TBB library directory" FORCE)
    set(TBB_DEFINITIONS "" CACHE STRING "TBB compiler definitions" FORCE)

    file(MAKE_DIRECTORY "${TBB_ROOT_DIR}/include")
    file(MAKE_DIRECTORY "${TBB_ROOT_DIR}/lib")

    foreach(c ${_tbb_components})
        set(_tbb_${c}_lib
            $<BUILD_INTERFACE:${TBB_ROOT_DIR}/lib/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}>
            $<INSTALL_INTERFACE:${c}>
        )

        # SOVERSION for libtbb: __TBB_BINARY_VERSION in external/onetbb/include/oneapi/tbb/version.h
        # SOVERSION for libtbbmalloc/libtbbmalloc_proxy: TBBMALLOC_BINARY_VERSION in external/onetbb/CMakeLists.txt
        if(${c} STREQUAL "tbb")
            set(_soversion 12)
        else()
            set(_soversion 2)
        endif()

        list(APPEND _tbb_libraries ${_tbb_${c}_lib})
        list(
            APPEND _tbb_build_byproducts
            "${TBB_ROOT_DIR}/lib/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${TBB_ROOT_DIR}/lib/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}.${_soversion}"
        )
    endforeach()
    unset(_soversion)

    set(TBB_LIBRARIES "${_tbb_libraries}" CACHE FILEPATH "TBB library files" FORCE)

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-tbb-build
        PREFIX ${TBB_ROOT_DIR}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/onetbb
        BINARY_DIR ${TBB_ROOT_DIR}/build
        BUILD_BYPRODUCTS ${_tbb_build_byproducts}
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${TBB_ROOT_DIR} -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_BUILD_RPATH=\$ORIGIN -DCMAKE_INSTALL_RPATH=\$ORIGIN -DTBB_TEST=OFF
            -DTBB_STRICT=OFF
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE} --target
            tbb tbbmalloc tbbmalloc_proxy
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
    )

    install(
        DIRECTORY ${TPL_STAGING_PREFIX}/tbb/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )
endif()

foreach(_DIR_TYPE INCLUDE LIBRARY)
    if(TBB_${_DIR_TYPE}_DIRS)
        list(REMOVE_DUPLICATES TBB_${_DIR_TYPE}_DIRS)
    endif()
endforeach()

if(NOT DEFINED _tbb_cmake_interface_definitions)
    set(_tbb_cmake_interface_definitions "")
endif()

target_include_directories(rocprofiler-systems-tbb SYSTEM INTERFACE ${TBB_INCLUDE_DIRS})
target_compile_definitions(
    rocprofiler-systems-tbb
    INTERFACE ${_tbb_cmake_interface_definitions}
)
target_link_directories(rocprofiler-systems-tbb INTERFACE ${TBB_LIBRARY_DIRS})
target_link_libraries(rocprofiler-systems-tbb INTERFACE ${TBB_LIBRARIES})

rocprofiler_systems_message(STATUS "TBB include directory: ${TBB_INCLUDE_DIRS}.")
rocprofiler_systems_message(STATUS "TBB library directory: ${TBB_LIBRARY_DIRS}.")
rocprofiler_systems_message(STATUS "TBB libraries: ${TBB_LIBRARIES}.")
rocprofiler_systems_message(STATUS "TBB definitions: ${TBB_DEFINITIONS}.")

# --------------------------------------------------------------------------------------#
# Create standard TBB::* targets for system packages only
# --------------------------------------------------------------------------------------#
# When using system packages, create standard targets that Dyninst's find_package(TBB)
# can discover.
if(NOT TARGET TBB::tbb AND NOT ROCPROFSYS_BUILD_TBB)
    # System package - create imported targets from found libraries
    rocprofiler_systems_message(
        STATUS
            "Creating TBB::* targets from system TBB (targets not provided by package)"
    )

    # Set TBB_ROOT for Dyninst's find_package(TBB) to use as a hint
    set(TBB_ROOT "${TBB_ROOT_DIR}" CACHE PATH "TBB root directory for Dyninst" FORCE)

    # Extract individual libraries from TBB_LIBRARIES list
    set(_tbb_lib "")
    set(_tbbmalloc_lib "")
    set(_tbbmalloc_proxy_lib "")

    foreach(_lib ${TBB_LIBRARIES})
        if(_lib MATCHES "libtbb\\.(so|a)")
            set(_tbb_lib "${_lib}")
        elseif(_lib MATCHES "libtbbmalloc_proxy\\.(so|a)")
            set(_tbbmalloc_proxy_lib "${_lib}")
        elseif(_lib MATCHES "libtbbmalloc\\.(so|a)")
            set(_tbbmalloc_lib "${_lib}")
        endif()
    endforeach()

    # Create TBB::tbb target
    if(_tbb_lib)
        add_library(TBB::tbb UNKNOWN IMPORTED)
        set_target_properties(
            TBB::tbb
            PROPERTIES
                IMPORTED_LOCATION "${_tbb_lib}"
                INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
        )
        if(_tbb_cmake_interface_definitions)
            set_target_properties(
                TBB::tbb
                PROPERTIES
                    INTERFACE_COMPILE_DEFINITIONS "${_tbb_cmake_interface_definitions}"
            )
        endif()
    endif()

    # Create TBB::tbbmalloc target
    if(_tbbmalloc_lib)
        add_library(TBB::tbbmalloc UNKNOWN IMPORTED)
        set_target_properties(
            TBB::tbbmalloc
            PROPERTIES
                IMPORTED_LOCATION "${_tbbmalloc_lib}"
                INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
        )
    endif()

    # Create TBB::tbbmalloc_proxy target
    if(_tbbmalloc_proxy_lib)
        add_library(TBB::tbbmalloc_proxy UNKNOWN IMPORTED)
        set_target_properties(
            TBB::tbbmalloc_proxy
            PROPERTIES
                IMPORTED_LOCATION "${_tbbmalloc_proxy_lib}"
                INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
        )
    endif()
endif()

# Create Dyninst::TBB for Dyninst when it is not already defined.

if(NOT TARGET Dyninst::TBB)
    add_library(Dyninst::TBB INTERFACE IMPORTED)

    if(TARGET TBB::tbb)
        target_link_libraries(Dyninst::TBB INTERFACE TBB::tbb)

        if(TARGET TBB::tbbmalloc)
            target_link_libraries(Dyninst::TBB INTERFACE TBB::tbbmalloc)
        endif()

        if(TARGET TBB::tbbmalloc_proxy)
            target_link_libraries(Dyninst::TBB INTERFACE TBB::tbbmalloc_proxy)
        endif()
    else()
        target_link_libraries(Dyninst::TBB INTERFACE ${TBB_LIBRARIES})
    endif()

    target_include_directories(Dyninst::TBB SYSTEM INTERFACE ${TBB_INCLUDE_DIRS})
    if(TBB_LIBRARY_DIRS)
        target_link_directories(Dyninst::TBB INTERFACE ${TBB_LIBRARY_DIRS})
    endif()
endif()
