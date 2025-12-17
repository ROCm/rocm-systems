# =====================================================================================
# ThreadingBuildingBlocks.cmake
#
# Configure Intel's Threading Building Blocks for Dyninst
#
# ----------------------------------------
#
# Accepts the following CMake variables
#
# TBB_ROOT_DIR        - Hint directory that contains the TBB installation TBB_INCLUDEDIR -
# Hint directory that contains the TBB headers files TBB_LIBRARYDIR      - Hint directory
# that contains the TBB library files TBB_LIBRARY         - Alias for TBB_LIBRARYDIR
# TBB_USE_DEBUG_BUILD - Use debug version of tbb libraries, if present TBB_MIN_VERSION -
# Minimum acceptable version of TBB
#
# Directly exports the following CMake variables
#
# TBB_ROOT_DIR        - Computed base directory of TBB installation TBB_INCLUDE_DIRS    -
# TBB include directory TBB_INCLUDE_DIR     - Alias for TBB_INCLUDE_DIRS TBB_LIBRARY_DIRS
# - TBB library directory TBB_LIBRARY_DIR - Alias for TBB_LIBRARY_DIRS TBB_DEFINITIONS -
# TBB compiler definitions TBB_LIBRARIES       - TBB library files
#
# TBB_<c>_LIBRARY_RELEASE - Path to the release version of component <c>
# TBB_<c>_LIBRARY_DEBUG   - Path to the debug version of component <c>
#
# NOTE: The exported TBB_ROOT_DIR can be different from the value provided by the user in
# the case that it is determined to build TBB from source. In such a case, TBB_ROOT_DIR
# will contain the directory of the from-source installation.
#
# See Modules/FindTBB.cmake for additional input and exported variables
#
# =====================================================================================

include_guard(GLOBAL)

if(TBB_FOUND)
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
    dyninst_message(
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
elseif(NOT ROCPROFSYS_BUILD_TBB)
    rocprofiler_systems_message(
        FATAL_ERROR
        "TBB was not found. Either configure cmake to find TBB properly or set ROCPROFSYS_BUILD_TBB=ON to download and build"
    )
else()
    rocprofiler_systems_add_cache_option(
        ROCPROFSYS_TBB_DOWNLOAD_VERSION "Version of TBB to download and install"
        STRING "2022.3.0"
    )

    rocprofiler_systems_message(
        STATUS "Attempting to build TBB(${ROCPROFSYS_TBB_DOWNLOAD_VERSION}) as external project"
    )

    set(TBB_ROOT_DIR ${TPL_STAGING_PREFIX}/tbb CACHE PATH "TBB root directory" FORCE)

    set(_tbb_libraries)
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

        list(APPEND _tbb_libraries ${_tbb_${c}_lib})
        list(
            APPEND _tbb_build_byproducts
            "${TBB_ROOT_DIR}/lib/lib${c}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
    endforeach()

    set(TBB_LIBRARIES "${_tbb_libraries}" CACHE FILEPATH "TBB library files" FORCE)

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-tbb-build
        PREFIX ${TBB_ROOT_DIR}
        URL
            https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${ROCPROFSYS_TBB_DOWNLOAD_VERSION}.tar.gz
        CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${TBB_ROOT_DIR} -DTBB_TEST=OFF
            -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_INCLUDEDIR=include
            -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath='$$ORIGIN'
        BUILD_BYPRODUCTS ${_tbb_build_byproducts}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    )

    if(CMAKE_STRIP)
        add_custom_command(
            TARGET rocprofiler-systems-tbb-build
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${STRIP_ARGS} ${TBB_ROOT_DIR}/lib/libtbb*.so*
            COMMENT "Stripping TBB libraries in ${TBB_ROOT_DIR}/lib ..."
        )
    endif()

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

target_include_directories(rocprofiler-systems-tbb SYSTEM INTERFACE ${TBB_INCLUDE_DIRS})
target_compile_definitions(rocprofiler-systems-tbb INTERFACE ${TBB_DEFINITIONS})
target_link_directories(rocprofiler-systems-tbb INTERFACE ${TBB_LIBRARY_DIRS})
target_link_libraries(rocprofiler-systems-tbb INTERFACE ${TBB_LIBRARIES})

rocprofiler_systems_message(STATUS "TBB include directory: ${TBB_INCLUDE_DIRS}.")
rocprofiler_systems_message(STATUS "TBB library directory: ${TBB_LIBRARY_DIRS}.")
rocprofiler_systems_message(STATUS "TBB libraries: ${TBB_LIBRARIES}.")
rocprofiler_systems_message(STATUS "TBB definitions: ${TBB_DEFINITIONS}.")
