# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

omnitrace_add_interface_library(
    omnitrace-headers "Provides minimal set of include flags to compile with omnitrace")
omnitrace_add_interface_library(omnitrace-threading "Enables multithreading support")
omnitrace_add_interface_library(
    omnitrace-dyninst
    "Provides flags and libraries for Dyninst (dynamic instrumentation)")
omnitrace_add_interface_library(omnitrace-hip "Provides flags and libraries for HIP")
omnitrace_add_interface_library(omnitrace-roctracer
                                "Provides flags and libraries for roctracer")
omnitrace_add_interface_library(omnitrace-rocprofiler
                                "Provides flags and libraries for rocprofiler")
omnitrace_add_interface_library(omnitrace-rocm-smi
                                "Provides flags and libraries for rocm-smi")
omnitrace_add_interface_library(omnitrace-mpi "Provides MPI or MPI headers")
omnitrace_add_interface_library(omnitrace-ptl "Enables PTL support (tasking)")
omnitrace_add_interface_library(omnitrace-papi "Enable PAPI support")
omnitrace_add_interface_library(omnitrace-ompt "Enable OMPT support")
omnitrace_add_interface_library(omnitrace-python "Enables Python support")
omnitrace_add_interface_library(omnitrace-perfetto "Enables Perfetto support")
omnitrace_add_interface_library(omnitrace-timemory "Provides timemory libraries")
omnitrace_add_interface_library(omnitrace-timemory-config
                                "CMake interface library applied to all timemory targets")
omnitrace_add_interface_library(omnitrace-compile-definitions "Compile definitions")

# libraries with relevant compile definitions
set(OMNITRACE_EXTENSION_LIBRARIES
    omnitrace::omnitrace-hip
    omnitrace::omnitrace-roctracer
    omnitrace::omnitrace-rocprofiler
    omnitrace::omnitrace-rocm-smi
    omnitrace::omnitrace-mpi
    omnitrace::omnitrace-ptl
    omnitrace::omnitrace-ompt
    omnitrace::omnitrace-papi
    omnitrace::omnitrace-perfetto)

target_include_directories(
    omnitrace-headers INTERFACE ${PROJECT_SOURCE_DIR}/source/lib/omnitrace
                                ${PROJECT_BINARY_DIR}/source/lib/omnitrace)

# include threading because of rooflines
target_link_libraries(omnitrace-headers INTERFACE omnitrace::omnitrace-threading)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

set(OMNITRACE_DEFAULT_ROCM_PATH
    /opt/rocm
    CACHE PATH "Default search path for ROCM")
if(EXISTS ${OMNITRACE_DEFAULT_ROCM_PATH})
    get_filename_component(_OMNITRACE_DEFAULT_ROCM_PATH "${OMNITRACE_DEFAULT_ROCM_PATH}"
                           REALPATH)

    if(NOT "${_OMNITRACE_DEFAULT_ROCM_PATH}" STREQUAL "${OMNITRACE_DEFAULT_ROCM_PATH}")
        set(OMNITRACE_DEFAULT_ROCM_PATH
            "${_OMNITRACE_DEFAULT_ROCM_PATH}"
            CACHE PATH "Default search path for ROCM" FORCE)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Threading
#
# ----------------------------------------------------------------------------------------#

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG OFF)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

if(pthread_LIBRARY)
    target_link_libraries(omnitrace-threading INTERFACE ${pthread_LIBRARY})
else()
    find_package(Threads ${omnitrace_FIND_QUIETLY} ${omnitrace_FIND_REQUIREMENT})
    if(Threads_FOUND)
        target_link_libraries(omnitrace-threading INTERFACE Threads::Threads)
    endif()
endif()

foreach(_LIB dl rt)
    find_library(${_LIB}_LIBRARY NAMES ${_LIB})
    find_package_handle_standard_args(${_LIB}-library REQUIRED_VARS ${_LIB}_LIBRARY)
    if(${_LIB}_LIBRARY)
        target_link_libraries(omnitrace-threading INTERFACE ${${_LIB}_LIBRARY})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# hip version
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_HIP
   OR OMNITRACE_USE_ROCTRACER
   OR OMNITRACE_USE_ROCPROFILER
   OR OMNITRACE_USE_ROCM_SMI)
    find_package(ROCmVersion)

    if(NOT ROCmVersion_FOUND)
        find_package(hip ${omnitrace_FIND_QUIETLY} REQUIRED HINTS
                     ${OMNITRACE_DEFAULT_ROCM_PATH} PATHS ${OMNITRACE_DEFAULT_ROCM_PATH})
        find_package(ROCmVersion REQUIRED HINTS ${ROCM_PATH} PATHS ${ROCM_PATH})
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR})

    set(OMNITRACE_ROCM_VERSION ${ROCmVersion_FULL_VERSION})
    set(OMNITRACE_HIP_VERSION_MAJOR ${ROCmVersion_MAJOR_VERSION})
    set(OMNITRACE_HIP_VERSION_MINOR ${ROCmVersion_MINOR_VERSION})
    set(OMNITRACE_HIP_VERSION_PATCH ${ROCmVersion_PATCH_VERSION})
    set(OMNITRACE_HIP_VERSION ${ROCmVersion_TRIPLE_VERSION})

    if(OMNITRACE_HIP_VERSION_MAJOR GREATER_EQUAL 4 AND OMNITRACE_HIP_VERSION_MINOR
                                                       GREATER 3)
        set(roctracer_kfdwrapper_LIBRARY)
    endif()

    if(NOT roctracer_kfdwrapper_LIBRARY)
        set(roctracer_kfdwrapper_LIBRARY)
    endif()

    omnitrace_add_feature(OMNITRACE_ROCM_VERSION "ROCm version used by omnitrace")
else()
    set(OMNITRACE_HIP_VERSION "0.0.0")
    set(OMNITRACE_HIP_VERSION_MAJOR 0)
    set(OMNITRACE_HIP_VERSION_MINOR 0)
    set(OMNITRACE_HIP_VERSION_PATCH 0)
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_HIP)
    find_package(hip ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-hip INTERFACE OMNITRACE_USE_HIP)
    target_link_libraries(omnitrace-hip INTERFACE hip::host)
endif()

# ----------------------------------------------------------------------------------------#
#
# roctracer
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_ROCTRACER)
    find_package(roctracer ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-roctracer
                                         INTERFACE OMNITRACE_USE_ROCTRACER)
    target_link_libraries(omnitrace-roctracer INTERFACE roctracer::roctracer
                                                        omnitrace::omnitrace-hip)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${roctracer_LIBRARY_DIRS}")
endif()

# ----------------------------------------------------------------------------------------#
#
# rocprofiler
#
# ----------------------------------------------------------------------------------------#
if(OMNITRACE_USE_ROCPROFILER)
    find_package(rocprofiler ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-rocprofiler
                                         INTERFACE OMNITRACE_USE_ROCPROFILER)
    target_link_libraries(omnitrace-rocprofiler INTERFACE rocprofiler::rocprofiler)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${rocprofiler_LIBRARY_DIRS}")
endif()

# ----------------------------------------------------------------------------------------#
#
# rocm-smi
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_ROCM_SMI)
    find_package(rocm-smi ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-rocm-smi
                                         INTERFACE OMNITRACE_USE_ROCM_SMI)
    target_link_libraries(omnitrace-rocm-smi INTERFACE rocm-smi::rocm-smi)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${rocm-smi_LIBRARY_DIRS}")
endif()

# ----------------------------------------------------------------------------------------#
#
# MPI
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_MPI)
    find_package(MPI ${omnitrace_FIND_QUIETLY} REQUIRED)
    target_link_libraries(omnitrace-mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
    omnitrace_target_compile_definitions(omnitrace-mpi INTERFACE TIMEMORY_USE_MPI=1
                                                                 OMNITRACE_USE_MPI)
elseif(OMNITRACE_USE_MPI_HEADERS)
    find_package(MPI-Headers ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(
        omnitrace-mpi INTERFACE TIMEMORY_USE_MPI_HEADERS=1 OMNITRACE_USE_MPI_HEADERS)
    target_link_libraries(omnitrace-mpi INTERFACE MPI::MPI_HEADERS)
endif()

# ----------------------------------------------------------------------------------------#
#
# OMPT
#
# ----------------------------------------------------------------------------------------#

omnitrace_target_compile_definitions(
    omnitrace-ompt INTERFACE OMNITRACE_USE_OMPT=$<BOOL:${OMNITRACE_USE_OMPT}>)

# ----------------------------------------------------------------------------------------#
#
# Dyninst
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_DYNINST)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/dyninst
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/dyninst.git
        REPO_BRANCH formatting)

    set(DYNINST_OPTION_PREFIX ON)
    set(DYNINST_BUILD_DOCS OFF)
    set(DYNINST_QUIET_CONFIG
        ON
        CACHE BOOL "Suppress dyninst cmake messages")
    set(DYNINST_BUILD_PARSE_THAT
        OFF
        CACHE BOOL "Build dyninst parseThat executable")
    set(DYNINST_BUILD_SHARED_LIBS
        ON
        CACHE BOOL "Build shared dyninst libraries")
    set(DYNINST_BUILD_STATIC_LIBS
        OFF
        CACHE BOOL "Build static dyninst libraries")
    set(DYNINST_ENABLE_LTO
        OFF
        CACHE BOOL "Enable LTO for dyninst libraries")

    omnitrace_save_variables(PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(DYNINST_TPL_INSTALL_PREFIX
        "omnitrace"
        CACHE PATH "Third-party library install-tree install prefix" FORCE)
    set(DYNINST_TPL_INSTALL_LIB_DIR
        "omnitrace"
        CACHE PATH "Third-party library install-tree install library prefix" FORCE)
    add_subdirectory(external/dyninst EXCLUDE_FROM_ALL)
    omnitrace_restore_variables(PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE)

    add_library(Dyninst::Dyninst INTERFACE IMPORTED)
    foreach(_LIB common dyninstAPI parseAPI instructionAPI symtabAPI stackwalk)
        target_link_libraries(Dyninst::Dyninst INTERFACE Dyninst::${_LIB})
    endforeach()

    foreach(
        _LIB
        common
        dynDwarf
        dynElf
        dyninstAPI
        dyninstAPI_RT
        instructionAPI
        parseAPI
        patchAPI
        pcontrol
        stackwalk
        symtabAPI)
        if(TARGET ${_LIB})
            install(
                TARGETS ${_LIB}
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
                COMPONENT dyninst
                PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace/include)
        endif()
    endforeach()

    omnitrace_install_tpl(dyninstAPI_RT omnitrace-rt
                          "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" core)

    # for packaging
    install(
        DIRECTORY ${DYNINST_TPL_STAGING_PREFIX}/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
        COMPONENT dyninst
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*")

    target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::Dyninst)

    set(OMNITRACE_DYNINST_API_RT
        ${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI_RT/libdyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX}
        )

    if(OMNITRACE_DYNINST_API_RT)
        omnitrace_target_compile_definitions(
            omnitrace-dyninst
            INTERFACE
                DYNINST_API_RT="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:$<TARGET_FILE_DIR:Dyninst::dyninstAPI_RT>:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:Dyninst::dyninstAPI_RT>:$<TARGET_FILE:Dyninst::dyninstAPI_RT>"
            )
    endif()

else()
    find_package(Dyninst ${omnitrace_FIND_QUIETLY} REQUIRED
                 COMPONENTS dyninstAPI parseAPI instructionAPI symtabAPI)

    if(TARGET Dyninst::Dyninst) # updated Dyninst CMake system was found
        # useful for defining the location of the runtime API
        find_library(
            OMNITRACE_DYNINST_API_RT dyninstAPI_RT
            HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
            PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
            PATH_SUFFIXES lib)

        if(OMNITRACE_DYNINST_API_RT)
            omnitrace_target_compile_definitions(
                omnitrace-dyninst INTERFACE DYNINST_API_RT="${OMNITRACE_DYNINST_API_RT}")
        endif()

        omnitrace_add_rpath(${Dyninst_LIBRARIES})
        target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::Dyninst)
    else() # updated Dyninst CMake system was not found
        set(_BOOST_COMPONENTS atomic system thread date_time)
        set(omnitrace_BOOST_COMPONENTS
            "${_BOOST_COMPONENTS}"
            CACHE STRING "Boost components used by Dyninst in omnitrace")
        set(Boost_NO_BOOST_CMAKE ON)
        find_package(Boost QUIET REQUIRED COMPONENTS ${omnitrace_BOOST_COMPONENTS})

        # some installs of dyninst don't set this properly
        if(EXISTS "${DYNINST_INCLUDE_DIR}" AND NOT DYNINST_HEADER_DIR)
            get_filename_component(DYNINST_HEADER_DIR "${DYNINST_INCLUDE_DIR}" REALPATH
                                   CACHE)
        else()
            find_path(
                DYNINST_HEADER_DIR
                NAMES BPatch.h dyninstAPI_RT.h
                HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATH_SUFFIXES include)
        endif()

        # useful for defining the location of the runtime API
        find_library(
            OMNITRACE_DYNINST_API_RT dyninstAPI_RT
            HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
            PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
            PATH_SUFFIXES lib)

        # try to find TBB
        find_package(TBB QUIET)

        # if fail try to use the Dyninst installed FindTBB.cmake
        if(NOT TBB_FOUND)
            list(APPEND CMAKE_MODULE_PATH ${Dyninst_DIR}/Modules)
            find_package(TBB QUIET)
        endif()

        if(NOT TBB_FOUND)
            find_path(
                TBB_INCLUDE_DIR
                NAMES tbb/tbb.h
                PATH_SUFFIXES include)
        endif()

        if(OMNITRACE_DYNINST_API_RT)
            omnitrace_target_compile_definitions(
                omnitrace-dyninst INTERFACE DYNINST_API_RT="${OMNITRACE_DYNINST_API_RT}")
        endif()

        if(Boost_DIR)
            get_filename_component(Boost_RPATH_DIR "${Boost_DIR}" DIRECTORY)
            get_filename_component(Boost_RPATH_DIR "${Boost_RPATH_DIR}" DIRECTORY)
            if(EXISTS "${Boost_RPATH_DIR}" AND IS_DIRECTORY "${Boost_RPATH_DIR}")
                set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${Boost_RPATH_DIR}")
            endif()
        endif()

        omnitrace_add_rpath(${DYNINST_LIBRARIES} ${Boost_LIBRARIES})
        target_link_libraries(omnitrace-dyninst INTERFACE ${DYNINST_LIBRARIES}
                                                          ${Boost_LIBRARIES})
        foreach(
            _TARG
            dyninst
            dyninstAPI
            instructionAPI
            symtabAPI
            parseAPI
            headers
            atomic
            system
            thread
            date_time
            TBB)
            if(TARGET Dyninst::${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::${_TARG})
            elseif(TARGET Boost::${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE Boost::${_TARG})
            elseif(TARGET ${_TARG})
                target_link_libraries(omnitrace-dyninst INTERFACE ${_TARG})
            endif()
        endforeach()
        target_include_directories(
            omnitrace-dyninst SYSTEM INTERFACE ${TBB_INCLUDE_DIR} ${Boost_INCLUDE_DIRS}
                                               ${DYNINST_HEADER_DIR})
        omnitrace_target_compile_definitions(omnitrace-dyninst
                                             INTERFACE OMNITRACE_USE_DYNINST)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Modify CMAKE_C_FLAGS and CMAKE_CXX_FLAGS with -static-libgcc and -static-libstdc++
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
    else()
        set(OMNITRACE_BUILD_STATIC_LIBGCC OFF)
    endif()
endif()

if(OMNITRACE_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_save_variables(STATIC_LIBSTDCXX_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
    else()
        set(OMNITRACE_BUILD_STATIC_LIBSTDCXX OFF)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Perfetto
#
# ----------------------------------------------------------------------------------------#

set(perfetto_DIR ${PROJECT_SOURCE_DIR}/external/perfetto)
omnitrace_checkout_git_submodule(
    RELATIVE_PATH external/perfetto
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://android.googlesource.com/platform/external/perfetto
    REPO_BRANCH v17.0
    TEST_FILE sdk/perfetto.cc)

include(Perfetto)

# ----------------------------------------------------------------------------------------#
#
# ELFIO
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_DEVICETRACE)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/elfio
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/ELFIO.git
        REPO_BRANCH set-offset-support)

    add_subdirectory(external/elfio)
endif()

# ----------------------------------------------------------------------------------------#
#
# papi submodule
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_PAPI AND OMNITRACE_BUILD_PAPI)
    include(PAPI)
endif()

# ----------------------------------------------------------------------------------------#
#
# timemory submodule
#
# ----------------------------------------------------------------------------------------#

target_compile_definitions(omnitrace-timemory-config INTERFACE TIMEMORY_PAPI_ARRAY_SIZE=16
                                                               TIMEMORY_USE_ROOFLINE=0)

set(TIMEMORY_EXTERNAL_INTERFACE_LIBRARY
    omnitrace-timemory-config
    CACHE STRING "timemory configuration interface library")
set(TIMEMORY_INSTALL_HEADERS
    OFF
    CACHE BOOL "Disable timemory header install")
set(TIMEMORY_INSTALL_CONFIG
    OFF
    CACHE BOOL "Disable timemory cmake configuration install")
set(TIMEMORY_INSTALL_LIBRARIES
    OFF
    CACHE BOOL "Disable timemory installation of libraries not needed at runtime")
set(TIMEMORY_INSTALL_ALL
    OFF
    CACHE BOOL "Disable install target depending on all target")
set(TIMEMORY_BUILD_C
    OFF
    CACHE BOOL "Disable timemory C library")
set(TIMEMORY_BUILD_FORTRAN
    OFF
    CACHE BOOL "Disable timemory Fortran library")
set(TIMEMORY_BUILD_TOOLS
    OFF
    CACHE BOOL "Ensure timem executable is built")
set(TIMEMORY_BUILD_EXCLUDE_FROM_ALL
    ON
    CACHE BOOL "Set timemory to only build dependencies")
set(TIMEMORY_BUILD_HIDDEN_VISIBILITY
    ON
    CACHE BOOL "Build timemory with hidden visibility")
set(TIMEMORY_QUIET_CONFIG
    ON
    CACHE BOOL "Make timemory configuration quieter")

# timemory feature settings
set(TIMEMORY_USE_MPI
    ${OMNITRACE_USE_MPI}
    CACHE BOOL "Enable MPI support in timemory" FORCE)
set(TIMEMORY_USE_GOTCHA
    ON
    CACHE BOOL "Enable GOTCHA support in timemory")
set(TIMEMORY_USE_PERFETTO
    OFF
    CACHE BOOL "Disable perfetto support in timemory")
set(TIMEMORY_USE_OMPT
    ${OMNITRACE_USE_OMPT}
    CACHE BOOL "Enable OMPT support in timemory" FORCE)
set(TIMEMORY_USE_PAPI
    ${OMNITRACE_USE_PAPI}
    CACHE BOOL "Enable PAPI support in timemory" FORCE)
set(TIMEMORY_USE_LIBUNWIND
    ON
    CACHE BOOL "Enable libunwind support in timemory")

if(DEFINED TIMEMORY_BUILD_GOTCHA AND NOT TIMEMORY_BUILD_GOTCHA)
    omnitrace_message(
        FATAL_ERROR
        "Using an external gotcha is not allowed due to known bug that has not been accepted upstream"
        )
endif()

# timemory feature build settings
set(TIMEMORY_BUILD_GOTCHA
    ON
    CACHE BOOL "Enable building GOTCHA library from submodule" FORCE)
set(TIMEMORY_BUILD_LIBUNWIND
    ${OMNITRACE_BUILD_LIBUNWIND}
    CACHE BOOL "Enable building libunwind library from submodule" FORCE)
set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    ${OMNITRACE_BUILD_EXTRA_OPTIMIZATIONS}
    CACHE BOOL "Enable building GOTCHA library from submodule" FORCE)

# timemory build settings
set(TIMEMORY_TLS_MODEL
    "global-dynamic"
    CACHE STRING "Thread-local static model" FORCE)
set(TIMEMORY_MAX_THREADS
    "${OMNITRACE_MAX_THREADS}"
    CACHE STRING "Max statically-allocated threads" FORCE)
set(TIMEMORY_SETTINGS_PREFIX
    "OMNITRACE_"
    CACHE STRING "Prefix used for settings and environment variables")
set(TIMEMORY_PROJECT_NAME
    "omnitrace"
    CACHE STRING "Name for configuration")
mark_as_advanced(TIMEMORY_SETTINGS_PREFIX)
mark_as_advanced(TIMEMORY_PROJECT_NAME)

omnitrace_checkout_git_submodule(
    RELATIVE_PATH external/timemory
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/NERSC/timemory.git
    REPO_BRANCH omnitrace)

omnitrace_save_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH)

# ensure timemory builds PIC static libs so that we don't have to install timemory shared
# lib
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TIMEMORY_CTP_OPTIONS GLOBAL)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # results in undefined symbols to component::base<T>::load()
    set(TIMEMORY_BUILD_HIDDEN_VISIBILITY
        OFF
        CACHE BOOL "" FORCE)
endif()

add_subdirectory(external/timemory EXCLUDE_FROM_ALL)

install(
    TARGETS gotcha
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
    COMPONENT gotcha)
if(OMNITRACE_BUILD_LIBUNWIND)
    install(
        DIRECTORY ${PROJECT_BINARY_DIR}/external/timemory/external/libunwind/install/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
        COMPONENT libunwind
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
endif()

omnitrace_restore_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH)

if(TARGET omnitrace-papi-build)
    foreach(_TARGET PAPI::papi timemory-core timemory-common timemory-papi-component
                    timemory-cxx)
        if(TARGET "${_TARGET}")
            add_dependencies(${_TARGET} omnitrace-papi-build)
        endif()
        foreach(_LINK shared static)
            if(TARGET "${_TARGET}-${_LINK}")
                add_dependencies(${_TARGET}-${_LINK} omnitrace-papi-build)
            endif()
        endforeach()
    endforeach()
endif()

target_link_libraries(
    omnitrace-timemory
    INTERFACE $<BUILD_INTERFACE:timemory::timemory-headers>
              $<BUILD_INTERFACE:timemory::timemory-gotcha>
              $<BUILD_INTERFACE:timemory::timemory-cxx-static>)

# ----------------------------------------------------------------------------------------#
#
# PTL (Parallel Tasking Library) submodule
#
# ----------------------------------------------------------------------------------------#

# timemory might provide PTL::ptl-shared
if(NOT TARGET PTL::ptl-shared)
    omnitrace_checkout_git_submodule(
        RELATIVE_PATH external/PTL
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/PTL.git
        REPO_BRANCH master)

    set(PTL_BUILD_EXAMPLES OFF)
    set(PTL_USE_TBB OFF)
    set(PTL_USE_GPU OFF)
    set(PTL_DEVELOPER_INSTALL OFF)

    if(NOT DEFINED BUILD_OBJECT_LIBS)
        set(BUILD_OBJECT_LIBS OFF)
    endif()
    omnitrace_save_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN)

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS OFF)
    set(BUILD_OBJECT_LIBS ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    add_subdirectory(external/PTL EXCLUDE_FROM_ALL)

    omnitrace_restore_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN)
endif()

target_sources(omnitrace-ptl INTERFACE $<TARGET_OBJECTS:PTL::ptl-object>)
target_link_libraries(omnitrace-ptl INTERFACE PTL::ptl-object)

# ----------------------------------------------------------------------------------------#
#
# Restore the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS in the inverse order
#
# ----------------------------------------------------------------------------------------#

# override compiler macros
include(Compilers)

if(OMNITRACE_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBSTDCXX_CXX VARIABLES CMAKE_CXX_FLAGS)
    endif()
endif()

if(OMNITRACE_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        omnitrace_restore_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
    endif()
endif()

omnitrace_add_feature(CMAKE_C_FLAGS "C compiler flags")
omnitrace_add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")

# ----------------------------------------------------------------------------------------#
#
# Python
#
# ----------------------------------------------------------------------------------------#

set(OMNITRACE_INSTALL_PYTHONDIR
    "${CMAKE_INSTALL_LIBDIR}/python/site-packages"
    CACHE STRING "Installation prefix for python")
set(CMAKE_INSTALL_PYTHONDIR ${OMNITRACE_INSTALL_PYTHONDIR})

if(OMNITRACE_USE_PYTHON)
    if(OMNITRACE_USE_PYTHON AND NOT OMNITRACE_BUILD_PYTHON)
        find_package(pybind11 REQUIRED)
    endif()

    include(ConfigPython)
endif()

# ----------------------------------------------------------------------------------------#
#
# Compile definitions
#
# ----------------------------------------------------------------------------------------#

foreach(_LIB ${OMNITRACE_EXTENSION_LIBRARIES})
    get_target_property(_COMPILE_DEFS ${_LIB} INTERFACE_COMPILE_DEFINITIONS)
    if(_COMPILE_DEFS)
        foreach(_DEF ${_COMPILE_DEFS})
            if("${_DEF}" MATCHES "OMNITRACE_")
                target_compile_definitions(omnitrace-compile-definitions
                                           INTERFACE ${_DEF})
            endif()
        endforeach()
    endif()
endforeach()
