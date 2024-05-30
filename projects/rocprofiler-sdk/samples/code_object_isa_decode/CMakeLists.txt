#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-samples-code-object-isa-decode LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(PkgConfig)

if(PkgConfig_FOUND)
    set(ENV{PKG_CONFIG_SYSTEM_INCLUDE_PATH} "")
    pkg_check_modules(DW libdw)

    if(DW_FOUND
       AND DW_INCLUDE_DIRS
       AND DW_LIBRARIES)
        set(libdw_INCLUDE_DIR
            "${DW_INCLUDE_DIRS}"
            CACHE FILEPATH "libdw include directory")
        set(libdw_LIBRARY
            "${DW_LIBRARIES}"
            CACHE FILEPATH "libdw libraries")
    endif()
endif()

if(NOT libdw_INCLUDE_DIR OR NOT libdw_LIBRARY)
    find_path(
        libdw_ROOT_DIR
        NAMES include/elfutils/libdw.h
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT})

    mark_as_advanced(libdw_ROOT_DIR)

    find_path(
        libdw_INCLUDE_DIR
        NAMES elfutils/libdw.h
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES include)

    find_library(
        libdw_LIBRARY
        NAMES dw
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES lib lib64)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libdw DEFAULT_MSG libdw_LIBRARY libdw_INCLUDE_DIR)

if(libdw_FOUND AND NOT TARGET libdw::libdw)
    add_library(libdw::libdw INTERFACE IMPORTED)
    if(TARGET PkgConfig::DW AND DW_FOUND)
        target_link_libraries(libdw::libdw INTERFACE PkgConfig::DW)
    else()
        target_link_libraries(libdw::libdw INTERFACE ${libdw_LIBRARY})
        target_include_directories(libdw::libdw SYSTEM INTERFACE ${libdw_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(libdw_INCLUDE_DIR libdw_LIBRARY)

find_package(rocprofiler-sdk REQUIRED)
find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/amd_comgr)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "-g")

add_executable(code-object-isa-decode)
target_sources(code-object-isa-decode PRIVATE main.cpp client.cpp)
target_link_libraries(
    code-object-isa-decode
    PRIVATE rocprofiler-sdk::samples-common-library rocprofiler-sdk::rocprofiler-sdk
            amd_comgr rocprofiler-sdk::samples-build-flags)

rocprofiler_samples_get_preload_env(PRELOAD_ENV code-object-isa-decode)

add_test(NAME code-object-isa-decode COMMAND $<TARGET_FILE:code-object-isa-decode>)

set_tests_properties(
    code-object-isa-decode
    PROPERTIES TIMEOUT 45 LABELS "samples" ENVIRONMENT "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION "threw an exception")
