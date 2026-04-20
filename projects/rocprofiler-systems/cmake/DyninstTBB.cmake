# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# =====================================================================================
# DyninstTBB.cmake
#
# Configure Threading Building Blocks (TBB) for Dyninst.
#
# ROCPROFSYS_BUILD_TBB=ON  → build oneTBB 2022.3.0 from external/onetbb submodule (always build Release)
# ROCPROFSYS_BUILD_TBB=OFF → use system TBB (e.g. libtbb-dev)
#
# Outputs (both paths):
#   TBB::tbb, TBB::tbbmalloc, TBB::tbbmalloc_proxy  — imported targets
#   Dyninst::TBB                                     — aggregated interface for Dyninst
#   rocprofiler-systems-tbb                          — interface target for this project
#   TBB_ROOT / TBB_ROOT_DIR                          — hint for Dyninst's own find_package
# =====================================================================================

include_guard(GLOBAL)

set(_tbb_components tbb tbbmalloc tbbmalloc_proxy)

if(ROCPROFSYS_BUILD_TBB)
    # =============================================================================
    # Bundled build: oneTBB 2022.3.0 from submodule
    # =============================================================================
    if(NOT UNIX)
        rocprofiler_systems_message(
            FATAL_ERROR "Building TBB from source is only supported on Unix"
        )
    endif()

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/onetbb
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/uxlfoundation/oneTBB.git
        REPO_BRANCH "v2022.3.0"
    )

    rocprofiler_systems_message(STATUS "Building TBB from external/onetbb submodule (Release)")

    # Always build/install bundled oneTBB as Release, independent of the main project's
    # CMAKE_BUILD_TYPE (preserves previous behavior).
    set(_tbb_bundled_build_type "Release")

    set(TBB_ROOT_DIR "${TPL_STAGING_PREFIX}/tbb" CACHE PATH "TBB root directory" FORCE)
    file(MAKE_DIRECTORY "${TBB_ROOT_DIR}/include" "${TBB_ROOT_DIR}/lib")

    # Build byproducts for Ninja dependency tracking.
    # SOVERSIONs from oneTBB 2022.3.0:
    #   libtbb:        __TBB_BINARY_VERSION  in include/oneapi/tbb/version.h  (= 12)
    #   libtbbmalloc*: TBBMALLOC_BINARY_VERSION  in CMakeLists.txt            (=  2)
    set(_tbb_build_byproducts
        "${TBB_ROOT_DIR}/lib/libtbb${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${TBB_ROOT_DIR}/lib/libtbb${CMAKE_SHARED_LIBRARY_SUFFIX}.12"
        "${TBB_ROOT_DIR}/lib/libtbbmalloc${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${TBB_ROOT_DIR}/lib/libtbbmalloc${CMAKE_SHARED_LIBRARY_SUFFIX}.2"
        "${TBB_ROOT_DIR}/lib/libtbbmalloc_proxy${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${TBB_ROOT_DIR}/lib/libtbbmalloc_proxy${CMAKE_SHARED_LIBRARY_SUFFIX}.2"
    )

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-tbb-build
        PREFIX "${TBB_ROOT_DIR}"
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/onetbb"
        BINARY_DIR "${TBB_ROOT_DIR}/build"
        BUILD_BYPRODUCTS ${_tbb_build_byproducts}
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=${_tbb_bundled_build_type}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${TBB_ROOT_DIR} -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_BUILD_RPATH=\$ORIGIN -DCMAKE_INSTALL_RPATH=\$ORIGIN -DTBB_TEST=OFF
            -DTBB_STRICT=OFF -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${_tbb_bundled_build_type}
            --target tbb tbbmalloc tbbmalloc_proxy
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${_tbb_bundled_build_type}
    )

    install(
        DIRECTORY "${TBB_ROOT_DIR}/lib/"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}"
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )

    # IMPORTED targets backed by ExternalProject output.
    # Build ordering is managed via external-prebuild in DyninstExternals.cmake.
    foreach(c ${_tbb_components})
        add_library(TBB::${c} UNKNOWN IMPORTED)
        set_target_properties(
            TBB::${c}
            PROPERTIES
                IMPORTED_LOCATION
                    "${TBB_ROOT_DIR}/lib/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                INTERFACE_INCLUDE_DIRECTORIES "${TBB_ROOT_DIR}/include"
        )
    endforeach()

    # FindTBB.cmake-compatible variables so other modules see a consistent TBB package
    # (avoids mixing stale cache entries from a prior system TBB configure).
    set(_tbb_bundled_incdir "${TBB_ROOT_DIR}/include")
    set(_tbb_bundled_libdir "${TBB_ROOT_DIR}/lib")
    set(_tbb_bundled_libs_release "")

    foreach(c ${_tbb_components})
        set(_tbb_lib "${_tbb_bundled_libdir}/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        list(APPEND _tbb_bundled_libs_release "${_tbb_lib}")
        set(TBB_${c}_LIBRARY_RELEASE
            "${_tbb_lib}"
            CACHE FILEPATH
            "Bundled oneTBB: ${c}"
            FORCE
        )
        set(TBB_${c}_FOUND TRUE CACHE BOOL "" FORCE)
    endforeach()

    set(TBB_INCLUDE_DIRS
        "${_tbb_bundled_incdir}"
        CACHE PATH
        "TBB include directories"
        FORCE
    )
    set(TBB_INCLUDE_DIR "${_tbb_bundled_incdir}" CACHE PATH "TBB include directory" FORCE)
    set(TBB_LIBRARY_DIRS
        "${_tbb_bundled_libdir}"
        CACHE PATH
        "TBB library directories"
        FORCE
    )
    set(TBB_LIBRARY
        "${_tbb_bundled_libdir}"
        CACHE PATH
        "TBB library directory hint"
        FORCE
    )
    set(TBB_LIBRARIES "${_tbb_bundled_libs_release}" CACHE STRING "TBB libraries" FORCE)
    set(TBB_DEFINITIONS "" CACHE STRING "TBB compile definitions" FORCE)
    set(TBB_VERSION "2022.3.0" CACHE STRING "Bundled oneTBB version" FORCE)
    set(TBB_FOUND TRUE CACHE BOOL "Bundled oneTBB is used" FORCE)
else()
    # =============================================================================
    # System package
    # =============================================================================
    set(TBB_USE_DEBUG_BUILD OFF CACHE BOOL "Use debug versions of TBB libraries")
    set(TBB_MIN_VERSION "2018.6" CACHE STRING "Minimum TBB version")
    set(TBB_ROOT_DIR "/usr" CACHE PATH "TBB root directory")
    set(TBB_LIBRARY "${TBB_ROOT_DIR}/lib")
    set(TBB_INCLUDE_DIR "${TBB_ROOT_DIR}/include")

    find_package(TBB ${TBB_MIN_VERSION} COMPONENTS ${_tbb_components})

    if(NOT TBB_FOUND)
        if(STERILE_BUILD)
            rocprofiler_systems_message(
                FATAL_ERROR
                    "TBB not found and cannot be downloaded because build is sterile"
            )
        else()
            rocprofiler_systems_message(
                FATAL_ERROR
                    "TBB not found. Install a system TBB package (e.g. libtbb-dev) or set ROCPROFSYS_BUILD_TBB=ON."
            )
        endif()
    endif()

    # Derive TBB_ROOT_DIR from the found location for Dyninst's find_package hint.
    if(TBB_INCLUDE_DIRS)
        list(GET TBB_INCLUDE_DIRS 0 _tbb_inc)
        get_filename_component(_tbb_root "${_tbb_inc}" DIRECTORY)
        set(TBB_ROOT_DIR "${_tbb_root}" CACHE PATH "TBB root directory" FORCE)
    elseif(TBB_DIR)
        string(REGEX REPLACE "/lib(/[^/]+)*/cmake/TBB[^/]*$" "" _tbb_root "${TBB_DIR}")
        if(NOT _tbb_root STREQUAL TBB_DIR)
            get_filename_component(_tbb_root "${_tbb_root}" ABSOLUTE)
            set(TBB_ROOT_DIR "${_tbb_root}" CACHE PATH "TBB root directory" FORCE)
        endif()
    endif()

    # Create TBB::* imported targets if find_package didn't provide them.
    if(NOT TARGET TBB::tbb)
        foreach(c ${_tbb_components})
            set(_lib "${TBB_${c}_LIBRARY_RELEASE}")
            if(NOT _lib)
                set(_lib "${TBB_${c}_LIBRARY_DEBUG}")
            endif()
            if(_lib)
                add_library(TBB::${c} UNKNOWN IMPORTED)
                set_target_properties(
                    TBB::${c}
                    PROPERTIES
                        IMPORTED_LOCATION "${_lib}"
                        INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
                )
            endif()
        endforeach()
    endif()
endif()

set(TBB_ROOT "${TBB_ROOT_DIR}" CACHE PATH "TBB root for Dyninst" FORCE)
rocprofiler_systems_message(STATUS "TBB include dirs: ${TBB_INCLUDE_DIRS}")
rocprofiler_systems_message(STATUS "TBB library dirs: ${TBB_LIBRARY_DIRS}")
rocprofiler_systems_message(STATUS "TBB libraries: ${TBB_LIBRARIES}")
rocprofiler_systems_message(STATUS "TBB version: ${TBB_VERSION}")

# =============================================================================
# Dyninst::TBB — aggregated target for Dyninst's consumption
# =============================================================================
if(NOT TARGET Dyninst::TBB)
    add_library(Dyninst::TBB INTERFACE IMPORTED)
    foreach(c ${_tbb_components})
        if(TARGET TBB::${c})
            target_link_libraries(Dyninst::TBB INTERFACE TBB::${c})
        endif()
    endforeach()
endif()

# =============================================================================
# rocprofiler-systems-tbb — this project's interface target
# =============================================================================
foreach(c ${_tbb_components})
    if(TARGET TBB::${c})
        target_link_libraries(rocprofiler-systems-tbb INTERFACE TBB::${c})
    endif()
endforeach()
