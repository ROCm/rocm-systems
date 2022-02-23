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
omnitrace_add_interface_library(omnitrace-rocm-smi
                                "Provides flags and libraries for rocm-smi")
omnitrace_add_interface_library(omnitrace-mpi "Provides MPI or MPI headers")
omnitrace_add_interface_library(omnitrace-ptl "Enables PTL support (tasking)")

target_include_directories(omnitrace-headers INTERFACE ${PROJECT_SOURCE_DIR}/include
                                                       ${PROJECT_BINARY_DIR}/include)

# include threading because of rooflines
target_link_libraries(omnitrace-headers INTERFACE omnitrace-threading)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

# ----------------------------------------------------------------------------------------#
#
# Threading
#
# ----------------------------------------------------------------------------------------#

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
endif()

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)
find_package(Threads ${omnitrace_FIND_QUIETLY} ${omnitrace_FIND_REQUIREMENT})

if(Threads_FOUND)
    target_link_libraries(omnitrace-threading INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif()

if(pthread_LIBRARY AND NOT WIN32)
    target_link_libraries(omnitrace-threading INTERFACE ${pthread_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
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
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(roctracer ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(omnitrace-roctracer
                                         INTERFACE OMNITRACE_USE_ROCTRACER)
    target_link_libraries(omnitrace-roctracer INTERFACE roctracer::roctracer
                                                        omnitrace::omnitrace-hip)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${roctracer_LIBRARY_DIRS}")
endif()

# ----------------------------------------------------------------------------------------#
#
# rocm-smmi
#
# ----------------------------------------------------------------------------------------#

if(OMNITRACE_USE_ROCM_SMI)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
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
elseif(OMNITRACE_USE_MPI_HEADERS)
    find_package(MPI-Headers ${omnitrace_FIND_QUIETLY} REQUIRED)
    omnitrace_target_compile_definitions(
        omnitrace-mpi INTERFACE TIMEMORY_USE_MPI_HEADERS OMNITRACE_USE_MPI_HEADERS)
    target_link_libraries(omnitrace-mpi INTERFACE MPI::MPI_HEADERS)
endif()

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
    add_subdirectory(external/dyninst)
    omnitrace_restore_variables(PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE)

    add_library(Dyninst::Dyninst INTERFACE IMPORTED)
    foreach(_LIB common dyninstAPI parseAPI instructionAPI symtabAPI stackwalk Boost TBB)
        target_link_libraries(Dyninst::Dyninst INTERFACE Dyninst::${_LIB})
    endforeach()

    target_link_libraries(omnitrace-dyninst INTERFACE Dyninst::Dyninst)

    set(OMNITRACE_DYNINST_API_RT
        ${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI_RT/libdyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX}
        )

    if(OMNITRACE_DYNINST_API_RT)
        omnitrace_target_compile_definitions(
            omnitrace-dyninst
            INTERFACE
                DYNINST_API_RT="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:$<TARGET_FILE_DIR:Dyninst::dyninstAPI_RT>:${CMAKE_INSTALL_PREFIX}/lib/$<TARGET_FILE_NAME:Dyninst::dyninstAPI_RT>:$<TARGET_FILE:Dyninst::dyninstAPI_RT>"
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
# timemory submodule
#
# ----------------------------------------------------------------------------------------#

set(TIMEMORY_INSTALL_HEADERS
    OFF
    CACHE BOOL "Disable timemory header install")
set(TIMEMORY_INSTALL_CONFIG
    OFF
    CACHE BOOL "Disable timemory cmake configuration install")
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
set(TIMEMORY_USE_LIBUNWIND
    ON
    CACHE BOOL "Enable libunwind support in timemory")

# timemory feature build settings
set(TIMEMORY_BUILD_GOTCHA
    ON
    CACHE BOOL "Enable building GOTCHA library from submodule")
set(TIMEMORY_BUILD_LIBUNWIND
    ON
    CACHE BOOL "Enable building libunwind library from submodule")
set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    ${OMNITRACE_BUILD_EXTRA_OPTIMIZATIONS}
    CACHE BOOL "Enable building GOTCHA library from submodule" FORCE)

# timemory build settings
set(TIMEMORY_TLS_MODEL
    "global-dynamic"
    CACHE STRING "Thread-local static model" FORCE)

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
    REPO_BRANCH gpu-kernel-instrumentation)

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

add_subdirectory(external/timemory)

omnitrace_restore_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH)

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

    omnitrace_save_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS CMAKE_POSITION_INDEPENDENT_CODE
                  CMAKE_CXX_VISIBILITY_PRESET CMAKE_VISIBILITY_INLINES_HIDDEN)

    set(BUILD_SHARED_LIBS ON)
    set(BUILD_STATIC_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    add_subdirectory(external/PTL)
    omnitrace_restore_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS CMAKE_POSITION_INDEPENDENT_CODE
                  CMAKE_CXX_VISIBILITY_PRESET CMAKE_VISIBILITY_INLINES_HIDDEN)
endif()

target_link_libraries(omnitrace-ptl INTERFACE PTL::ptl-shared)
