# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

rocprofiler_systems_add_interface_library(
    rocprofiler-systems-headers
    "Provides minimal set of include flags to compile with rocprofiler-systems"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-threading
                                          "Enables multithreading support"
)
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-dyninst
    "Provides flags and libraries for Dyninst (dynamic instrumentation)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-boost
                                          "Boost interface library (for Dyninst)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-elfutils
                                          "ElfUtils interface library (for Dyninst)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-libiberty
                                          "LibIberty interface library (for Dyninst)"
)
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-tbb "Threading Building Blocks interface library (for Dyninst)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-rocm
                                          "Provides flags and libraries for ROCm"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-mpi
                                          "Provides MPI or MPI headers"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-libva
                                          "Provides VA-API headers"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-bfd
                                          "Provides Binary File Descriptor (BFD)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-ptl
                                          "Enables PTL support (tasking)"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-papi "Enable PAPI support")
rocprofiler_systems_add_interface_library(rocprofiler-systems-ompt "Enable OMPT support")
rocprofiler_systems_add_interface_library(rocprofiler-systems-python
                                          "Enables Python support"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-perfetto
                                          "Enables Perfetto support"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-sqlite3
                                          "Use SQLite3 for rocpd data storage"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-timemory
                                          "Provides timemory libraries"
)
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-timemory-config
    "CMake interface library applied to all timemory targets"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-compile-definitions
                                          "Compile definitions"
)

# libraries with relevant compile definitions
set(ROCPROFSYS_EXTENSION_LIBRARIES
    rocprofiler-systems::rocprofiler-systems-rocm
    rocprofiler-systems::rocprofiler-systems-bfd
    rocprofiler-systems::rocprofiler-systems-mpi
    rocprofiler-systems::rocprofiler-systems-ptl
    rocprofiler-systems::rocprofiler-systems-ompt
    rocprofiler-systems::rocprofiler-systems-papi
    rocprofiler-systems::rocprofiler-systems-perfetto
)

target_include_directories(
    rocprofiler-systems-headers
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/lib>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/lib/core>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys-dl>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys-user>
)

# include threading because of rooflines
target_link_libraries(
    rocprofiler-systems-headers
    INTERFACE rocprofiler-systems::rocprofiler-systems-threading
)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

set(ROCPROFSYS_DEFAULT_ROCM_PATH /opt/rocm CACHE PATH "Default search path for ROCM")
if(EXISTS ${ROCPROFSYS_DEFAULT_ROCM_PATH})
    get_filename_component(
        _ROCPROFSYS_DEFAULT_ROCM_PATH
        "${ROCPROFSYS_DEFAULT_ROCM_PATH}"
        REALPATH
    )

    if(NOT "${_ROCPROFSYS_DEFAULT_ROCM_PATH}" STREQUAL "${ROCPROFSYS_DEFAULT_ROCM_PATH}")
        set(ROCPROFSYS_DEFAULT_ROCM_PATH
            "${_ROCPROFSYS_DEFAULT_ROCM_PATH}"
            CACHE PATH
            "Default search path for ROCM"
            FORCE
        )
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
    target_link_libraries(rocprofiler-systems-threading INTERFACE ${pthread_LIBRARY})
else()
    find_package(
        Threads
        ${rocprofiler_systems_FIND_QUIETLY}
        ${rocprofiler_systems_FIND_REQUIREMENT}
    )
    if(Threads_FOUND)
        target_link_libraries(rocprofiler-systems-threading INTERFACE Threads::Threads)
    endif()
endif()

foreach(_LIB dl rt)
    find_library(${_LIB}_LIBRARY NAMES ${_LIB})
    find_package_handle_standard_args(${_LIB}-library REQUIRED_VARS ${_LIB}_LIBRARY)
    if(${_LIB}_LIBRARY)
        target_link_libraries(rocprofiler-systems-threading INTERFACE ${${_LIB}_LIBRARY})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# ROCm Version
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_USE_ROCM)
    find_package(ROCmVersion)

    if(NOT ROCmVersion_FOUND)
        find_package(
            hip
            ${rocprofiler_systems_FIND_QUIETLY}
            REQUIRED
            HINTS ${ROCPROFSYS_DEFAULT_ROCM_PATH}
            PATHS ${ROCPROFSYS_DEFAULT_ROCM_PATH}
        )
        if(SPACK_BUILD)
            find_package(ROCmVersion HINTS ${ROCM_PATH} PATHS ${ROCM_PATH})
        else()
            find_package(ROCmVersion REQUIRED HINTS ${ROCM_PATH} PATHS ${ROCM_PATH})
        endif()
    endif()

    if(NOT ROCmVersion_FOUND)
        rocm_version_compute("${hip_VERSION}" _local)

        foreach(_V ${ROCmVersion_VARIABLES})
            set(_CACHE_VAR ROCmVersion_${_V}_VERSION)
            set(_LOCAL_VAR _local_${_V}_VERSION)
            set(ROCmVersion_${_V}_VERSION
                "${${_LOCAL_VAR}}"
                CACHE STRING
                "ROCm ${_V} version"
            )
            rocm_version_watch_for_change(${_CACHE_VAR})
        endforeach()
    else()
        list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR})
    endif()

    set(ROCPROFSYS_ROCM_VERSION ${ROCmVersion_FULL_VERSION})
    set(ROCPROFSYS_ROCM_VERSION_MAJOR ${ROCmVersion_MAJOR_VERSION})
    set(ROCPROFSYS_ROCM_VERSION_MINOR ${ROCmVersion_MINOR_VERSION})
    set(ROCPROFSYS_ROCM_VERSION_PATCH ${ROCmVersion_PATCH_VERSION})
    set(ROCPROFSYS_ROCM_VERSION ${ROCmVersion_TRIPLE_VERSION})

    rocprofiler_systems_add_feature(ROCPROFSYS_ROCM_VERSION
                                    "ROCm version used by rocprofiler-systems"
    )
else()
    set(ROCPROFSYS_ROCM_VERSION "0.0.0")
    set(ROCPROFSYS_ROCM_VERSION_MAJOR 0)
    set(ROCPROFSYS_ROCM_VERSION_MINOR 0)
    set(ROCPROFSYS_ROCM_VERSION_PATCH 0)
endif()

# ----------------------------------------------------------------------------------------#
#
# ROCm
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_USE_ROCM)
    find_package(rocprofiler-sdk ${rocprofiler_systems_FIND_QUIETLY} REQUIRED)
    rocprofiler_systems_target_compile_definitions(rocprofiler-systems-rocm
                                                   INTERFACE ROCPROFSYS_USE_ROCM
    )
    target_link_libraries(
        rocprofiler-systems-rocm
        INTERFACE rocprofiler-sdk::rocprofiler-sdk
    )

    find_package(amd-smi ${rocprofiler_systems_FIND_QUIETLY} REQUIRED)
    target_link_libraries(rocprofiler-systems-rocm INTERFACE amd-smi::amd-smi)
endif()

# ----------------------------------------------------------------------------------------#
#
# MPI
#
# ----------------------------------------------------------------------------------------#

# suppress warning during CI that MPI_HEADERS_ALLOW_MPICH was unused
set(_ROCPROFSYS_MPI_HEADERS_ALLOW_MPICH ${MPI_HEADERS_ALLOW_MPICH})

if(ROCPROFSYS_USE_MPI)
    find_package(MPI ${rocprofiler_systems_FIND_QUIETLY} REQUIRED)
    target_link_libraries(rocprofiler-systems-mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
    rocprofiler_systems_target_compile_definitions(rocprofiler-systems-mpi
                                                   INTERFACE ROCPROFSYS_USE_MPI
    )
elseif(ROCPROFSYS_USE_MPI_HEADERS)
    find_package(MPI-Headers ${rocprofiler_systems_FIND_QUIETLY} REQUIRED)
    rocprofiler_systems_target_compile_definitions(rocprofiler-systems-mpi
                                                   INTERFACE ROCPROFSYS_USE_MPI_HEADERS
    )
    target_link_libraries(rocprofiler-systems-mpi INTERFACE MPI::MPI_HEADERS)
endif()

# ----------------------------------------------------------------------------------------#
#
# OMPT
#
# ----------------------------------------------------------------------------------------#

rocprofiler_systems_target_compile_definitions(
    rocprofiler-systems-ompt INTERFACE ROCPROFSYS_USE_OMPT=$<BOOL:${ROCPROFSYS_USE_OMPT}>
)

# ----------------------------------------------------------------------------------------#
#
# Dyninst
#
# ----------------------------------------------------------------------------------------#
include(DyninstExternals)
if(ROCPROFSYS_BUILD_DYNINST)
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/dyninst
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/ROCm/dyninst.git
        REPO_BRANCH dyninst_13
    )
message(WARNING "########## CURRENT PROJECT DIRECTORY --------> ${PROJECT_SOURCE_DIR} #############")
    set(DYNINST_OPTION_PREFIX ON)
    set(DYNINST_BUILD_DOCS OFF)
    set(DYNINST_BUILD_RTLIB OFF)
    set(DYNINST_QUIET_CONFIG ON CACHE BOOL "Suppress dyninst cmake messages")
    set(DYNINST_BUILD_PARSE_THAT OFF CACHE BOOL "Build dyninst parseThat executable")
    set(DYNINST_BUILD_SHARED_LIBS ON CACHE BOOL "Build shared dyninst libraries")
    set(DYNINST_BUILD_STATIC_LIBS OFF CACHE BOOL "Build static dyninst libraries")
    set(DYNINST_ENABLE_LTO OFF CACHE BOOL "Enable LTO for dyninst libraries")

    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH "")
    endif()

    if(NOT DEFINED CMAKE_BUILD_RPATH)
        set(CMAKE_BUILD_RPATH "")
    endif()

    rocprofiler_systems_save_variables(
        PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE CMAKE_INSTALL_RPATH
                      CMAKE_BUILD_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

    set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/${PROJECT_NAME}")
    set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${PROJECT_NAME}")
    set(DYNINST_TPL_INSTALL_PREFIX
        "${PROJECT_NAME}"
        CACHE PATH
        "Third-party library install-tree install prefix"
        FORCE
    )
    set(DYNINST_TPL_INSTALL_LIB_DIR
        "${PROJECT_NAME}"
        CACHE PATH
        "Third-party library install-tree install library prefix"
        FORCE
    )

    add_subdirectory(external/dyninst EXCLUDE_FROM_ALL)
    rocprofiler_systems_restore_variables(
        PIC VARIABLES CMAKE_POSITION_INDEPENDENT_CODE CMAKE_INSTALL_RPATH
                      CMAKE_BUILD_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
    )

    add_library(Dyninst::Dyninst INTERFACE IMPORTED)
    foreach(
        _LIB
        common
        dyninstAPI
        parseAPI
        instructionAPI
        symtabAPI
        stackwalk
    )
        target_link_libraries(Dyninst::Dyninst INTERFACE Dyninst::${_LIB})
    endforeach()

    foreach(
        _LIB
        common
        dynDwarf
        dynElf
        dyninstAPI
        instructionAPI
        parseAPI
        patchAPI
        pcontrol
        stackwalk
        symtabAPI
    )
        if(TARGET ${_LIB})
            install(
                TARGETS ${_LIB}
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                COMPONENT dyninst
                PUBLIC_HEADER
                    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/dyninst
            )
        endif()
    endforeach()

    foreach(
        _LIB
        common
        dynDwarf
        dynElf
        dyninstAPI
        instructionAPI
        parseAPI
        patchAPI
        pcontrol
        stackwalk
        symtabAPI
    )
        if(TARGET ${_LIB})
            add_dependencies(${_LIB} external-prebuild)
            if(NOT TARGET Dyninst::${_LIB})
                add_library(Dyninst::${_LIB} ALIAS ${_LIB})
            endif()
        endif()
    endforeach()

    target_link_libraries(rocprofiler-systems-dyninst INTERFACE Dyninst::Dyninst)
else()
    # Find Boost before finding Dyninst
    find_package(Boost)
    if(NOT TARGET Dyninst::Boost_headers)
        add_library(Dyninst::Boost_headers INTERFACE IMPORTED)
        target_include_directories(
            Dyninst::Boost_headers
            SYSTEM
            INTERFACE ${Boost_INCLUDE_DIRS}
        )
    endif()

    find_package(
        Dyninst
        ${rocprofiler_systems_FIND_QUIETLY}
        REQUIRED
        COMPONENTS dyninstAPI parseAPI instructionAPI symtabAPI
    )

    if(TARGET Dyninst::Dyninst) # updated Dyninst CMake system was found
        target_link_libraries(rocprofiler-systems-dyninst INTERFACE Dyninst::Dyninst)
    else() # updated Dyninst CMake system was not found
        set(_BOOST_COMPONENTS atomic system thread date_time)
        set(rocprofiler_systems_BOOST_COMPONENTS
            "${_BOOST_COMPONENTS}"
            CACHE STRING
            "Boost components used by Dyninst in rocprofiler-systems"
        )
        set(Boost_NO_BOOST_CMAKE ON)
        find_package(
            Boost
            QUIET
            REQUIRED
            COMPONENTS ${rocprofiler_systems_BOOST_COMPONENTS}
        )

        # some installs of dyninst don't set this properly
        if(EXISTS "${DYNINST_INCLUDE_DIR}" AND NOT DYNINST_HEADER_DIR)
            get_filename_component(
                DYNINST_HEADER_DIR
                "${DYNINST_INCLUDE_DIR}"
                REALPATH
                CACHE
            )
        else()
            find_path(
                DYNINST_HEADER_DIR
                NAMES BPatch.h dyninstAPI_RT.h
                HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
                PATH_SUFFIXES include
            )
        endif()

        # try to find TBB
        find_package(TBB QUIET)

        # if fail try to use the Dyninst installed FindTBB.cmake
        if(NOT TBB_FOUND)
            list(APPEND CMAKE_MODULE_PATH ${Dyninst_DIR}/Modules)
            find_package(TBB QUIET)
        endif()

        if(NOT TBB_FOUND)
            find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h PATH_SUFFIXES include)
        endif()

        target_link_libraries(
            rocprofiler-systems-dyninst
            INTERFACE ${DYNINST_LIBRARIES} ${Boost_LIBRARIES}
        )
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
            TBB
        )
            if(TARGET Dyninst::${_TARG})
                target_link_libraries(
                    rocprofiler-systems-dyninst
                    INTERFACE Dyninst::${_TARG}
                )
            elseif(TARGET Boost::${_TARG})
                target_link_libraries(
                    rocprofiler-systems-dyninst
                    INTERFACE Boost::${_TARG}
                )
            elseif(TARGET ${_TARG})
                target_link_libraries(rocprofiler-systems-dyninst INTERFACE ${_TARG})
            endif()
        endforeach()
        target_include_directories(
            rocprofiler-systems-dyninst
            SYSTEM
            INTERFACE ${TBB_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${DYNINST_HEADER_DIR}
        )
        rocprofiler_systems_target_compile_definitions(rocprofiler-systems-dyninst
                                                       INTERFACE ROCPROFSYS_USE_DYNINST
        )
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Modify CMAKE_C_FLAGS and CMAKE_CXX_FLAGS with -static-libgcc and -static-libstdc++
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_save_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_save_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
    else()
        set(ROCPROFSYS_BUILD_STATIC_LIBGCC OFF)
    endif()
endif()

if(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_save_variables(STATIC_LIBSTDCXX_CXX VARIABLES CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
    else()
        set(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX OFF)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Perfetto
#
# ----------------------------------------------------------------------------------------#

set(perfetto_DIR ${PROJECT_SOURCE_DIR}/external/perfetto)
rocprofiler_systems_checkout_git_submodule(
    RELATIVE_PATH external/perfetto
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/google/perfetto.git
    REPO_BRANCH v46.0
    TEST_FILE sdk/perfetto.cc
)

include(Perfetto)

# ----------------------------------------------------------------------------------------#
#
# SQLite3
#
# ----------------------------------------------------------------------------------------#

include(SQLite3)

# ----------------------------------------------------------------------------------------#
#
# ELFIO
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_BUILD_DEVICETRACE)
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/elfio
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/ELFIO.git
        REPO_BRANCH set-offset-support
    )

    add_subdirectory(external/elfio)
endif()

# ----------------------------------------------------------------------------------------#
#
# papi submodule
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_USE_PAPI AND ROCPROFSYS_BUILD_PAPI)
    include(PAPI)
endif()

# ----------------------------------------------------------------------------------------#
#
# timemory submodule
#
# ----------------------------------------------------------------------------------------#

target_compile_definitions(
    rocprofiler-systems-timemory-config
    INTERFACE
        TIMEMORY_PAPI_ARRAY_SIZE=12
        TIMEMORY_USE_ROOFLINE=0
        TIMEMORY_USE_ERT=0
        TIMEMORY_USE_CONTAINERS=0
        TIMEMORY_USE_ERT_EXTERN=0
        TIMEMORY_USE_CONTAINERS_EXTERN=0
)

if(ROCPROFSYS_BUILD_STACK_PROTECTOR)
    add_target_flag_if_avail(rocprofiler-systems-timemory-config
                             "-fstack-protector-strong" "-Wstack-protector"
    )
endif()

if(ROCPROFSYS_BUILD_DEBUG)
    add_target_flag_if_avail(rocprofiler-systems-timemory-config
                             "-fno-omit-frame-pointer" "-g3"
    )
endif()

set(TIMEMORY_EXTERNAL_INTERFACE_LIBRARY
    rocprofiler-systems-timemory-config
    CACHE STRING
    "timemory configuration interface library"
)
set(TIMEMORY_INSTALL_HEADERS OFF CACHE BOOL "Disable timemory header install")
set(TIMEMORY_INSTALL_CONFIG OFF CACHE BOOL "Disable timemory cmake configuration install")
set(TIMEMORY_INSTALL_LIBRARIES
    OFF
    CACHE BOOL
    "Disable timemory installation of libraries not needed at runtime"
)
set(TIMEMORY_INSTALL_ALL OFF CACHE BOOL "Disable install target depending on all target")
set(TIMEMORY_BUILD_C OFF CACHE BOOL "Disable timemory C library")
set(TIMEMORY_BUILD_FORTRAN OFF CACHE BOOL "Disable timemory Fortran library")
set(TIMEMORY_BUILD_TOOLS OFF CACHE BOOL "Ensure timem executable is built")
set(TIMEMORY_BUILD_EXCLUDE_FROM_ALL
    ON
    CACHE BOOL
    "Set timemory to only build dependencies"
)
set(TIMEMORY_BUILD_HIDDEN_VISIBILITY
    ON
    CACHE BOOL
    "Build timemory with hidden visibility"
)
set(TIMEMORY_QUIET_CONFIG ON CACHE BOOL "Make timemory configuration quieter")

# timemory feature settings
set(TIMEMORY_USE_GOTCHA ON CACHE BOOL "Enable GOTCHA support in timemory")
set(TIMEMORY_USE_PERFETTO OFF CACHE BOOL "Disable perfetto support in timemory")
set(TIMEMORY_USE_OMPT
    ${ROCPROFSYS_USE_OMPT}
    CACHE BOOL
    "Enable OMPT support in timemory"
    FORCE
)
set(TIMEMORY_USE_PAPI
    ${ROCPROFSYS_USE_PAPI}
    CACHE BOOL
    "Enable PAPI support in timemory"
    FORCE
)
set(TIMEMORY_USE_BFD
    ${ROCPROFSYS_USE_BFD}
    CACHE BOOL
    "Enable BFD support in timemory"
    FORCE
)
set(TIMEMORY_USE_LIBUNWIND ON CACHE BOOL "Enable libunwind support in timemory")
set(TIMEMORY_USE_VISIBILITY OFF CACHE BOOL "Enable/disable using visibility decorations")
set(TIMEMORY_USE_SANITIZER
    ${ROCPROFSYS_USE_SANITIZER}
    CACHE BOOL
    "Build with -fsanitze=\${ROCPROFSYS_SANITIZER_TYPE}"
    FORCE
)
set(TIMEMORY_SANITIZER_TYPE
    ${ROCPROFSYS_SANITIZER_TYPE}
    CACHE STRING
    "Sanitizer type, e.g. leak, thread, address, memory, etc."
    FORCE
)

if(DEFINED TIMEMORY_BUILD_GOTCHA AND NOT TIMEMORY_BUILD_GOTCHA)
    rocprofiler_systems_message(
        FATAL_ERROR
        "Using an external gotcha is not allowed due to known bug that has not been accepted upstream"
    )
endif()

# timemory feature build settings
set(TIMEMORY_BUILD_GOTCHA
    ON
    CACHE BOOL
    "Enable building GOTCHA library from submodule"
    FORCE
)
set(TIMEMORY_BUILD_LIBUNWIND
    ${ROCPROFSYS_BUILD_LIBUNWIND}
    CACHE BOOL
    "Enable building libunwind library from submodule"
    FORCE
)
set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    ${ROCPROFSYS_BUILD_EXTRA_OPTIMIZATIONS}
    CACHE BOOL
    "Enable building GOTCHA library from submodule"
    FORCE
)
set(TIMEMORY_BUILD_ERT OFF CACHE BOOL "Disable building ERT support" FORCE)
set(TIMEMORY_BUILD_CONTAINERS
    OFF
    CACHE BOOL
    "Disable building container extern templates (unused)"
    FORCE
)

# timemory build settings
set(TIMEMORY_TLS_MODEL "global-dynamic" CACHE STRING "Thread-local static model" FORCE)
set(TIMEMORY_MAX_THREADS
    "${ROCPROFSYS_MAX_THREADS}"
    CACHE STRING
    "Max statically-allocated threads"
    FORCE
)
set(TIMEMORY_SETTINGS_PREFIX
    "ROCPROFSYS_"
    CACHE STRING
    "Prefix used for settings and environment variables"
)
set(TIMEMORY_PROJECT_NAME "rocprofiler-systems" CACHE STRING "Name for configuration")
set(TIMEMORY_CXX_LIBRARY_EXCLUDE
    "kokkosp.cpp;pthread.cpp;timemory_c.cpp;trace.cpp;weak.cpp;library.cpp"
    CACHE STRING
    "Timemory C++ library implementation files to exclude from compiling"
)

mark_as_advanced(TIMEMORY_SETTINGS_PREFIX)
mark_as_advanced(TIMEMORY_PROJECT_NAME)

rocprofiler_systems_checkout_git_submodule(
    RELATIVE_PATH external/timemory
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/ROCm/timemory.git
    REPO_BRANCH omnitrace
)

rocprofiler_systems_save_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH
)

# ensure timemory builds PIC static libs so that we don't have to install timemory shared
# lib
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TIMEMORY_CTP_OPTIONS GLOBAL)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # results in undefined symbols to component::base<T>::load()
    set(TIMEMORY_BUILD_HIDDEN_VISIBILITY OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(external/timemory EXCLUDE_FROM_ALL)

install(
    TARGETS gotcha
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    COMPONENT gotcha
)
if(ROCPROFSYS_BUILD_LIBUNWIND)
    install(
        DIRECTORY ${PROJECT_BINARY_DIR}/external/timemory/external/libunwind/install/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        COMPONENT libunwind
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )
endif()

rocprofiler_systems_restore_variables(
    BUILD_CONFIG VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS
                           CMAKE_POSITION_INDEPENDENT_CODE CMAKE_PREFIX_PATH
)

if(TARGET rocprofiler-systems-papi-build)
    foreach(
        _TARGET
        PAPI::papi
        timemory-core
        timemory-common
        timemory-papi-component
        timemory-cxx
    )
        if(TARGET "${_TARGET}")
            add_dependencies(${_TARGET} rocprofiler-systems-papi-build)
        endif()
        foreach(_LINK shared static)
            if(TARGET "${_TARGET}-${_LINK}")
                add_dependencies(${_TARGET}-${_LINK} rocprofiler-systems-papi-build)
            endif()
        endforeach()
    endforeach()
endif()

target_link_libraries(
    rocprofiler-systems-timemory
    INTERFACE
        $<BUILD_INTERFACE:timemory::timemory-headers>
        $<BUILD_INTERFACE:timemory::timemory-gotcha>
        $<BUILD_INTERFACE:timemory::timemory-cxx-static>
)

target_link_libraries(
    rocprofiler-systems-bfd
    INTERFACE $<BUILD_INTERFACE:timemory::timemory-bfd>
)

if(ROCPROFSYS_USE_BFD)
    rocprofiler_systems_target_compile_definitions(rocprofiler-systems-bfd
                                                   INTERFACE ROCPROFSYS_USE_BFD
    )
endif()

find_package(Libva-headers ${rocprofiler_systems_FIND_QUIETLY} REQUIRED)
target_include_directories(
    rocprofiler-systems-libva
    INTERFACE ${LIBVA_HEADERS_INCLUDE_DIR}
)

# ----------------------------------------------------------------------------------------#
#
# PTL (Parallel Tasking Library) submodule
#
# ----------------------------------------------------------------------------------------#

# timemory might provide PTL::ptl-shared
if(NOT TARGET PTL::ptl-shared)
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/PTL
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/PTL.git
        REPO_BRANCH omnitrace
    )

    set(PTL_BUILD_EXAMPLES OFF)
    set(PTL_USE_TBB OFF)
    set(PTL_USE_GPU OFF)
    set(PTL_DEVELOPER_INSTALL OFF)

    if(NOT DEFINED BUILD_OBJECT_LIBS)
        set(BUILD_OBJECT_LIBS OFF)
    endif()
    rocprofiler_systems_save_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN
    )

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS OFF)
    set(BUILD_OBJECT_LIBS ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    add_subdirectory(external/PTL EXCLUDE_FROM_ALL)

    rocprofiler_systems_restore_variables(
        BUILD_CONFIG
        VARIABLES BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_OBJECT_LIBS
                  CMAKE_POSITION_INDEPENDENT_CODE CMAKE_CXX_VISIBILITY_PRESET
                  CMAKE_VISIBILITY_INLINES_HIDDEN
    )
endif()

target_sources(
    rocprofiler-systems-ptl
    INTERFACE $<BUILD_INTERFACE:$<TARGET_OBJECTS:PTL::ptl-object>>
)
target_include_directories(
    rocprofiler-systems-ptl
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/PTL/source>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/PTL/source>
)

# ----------------------------------------------------------------------------------------#
#
# Restore the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS in the inverse order
#
# ----------------------------------------------------------------------------------------#

# override compiler macros
include(Compilers)

if(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_restore_variables(STATIC_LIBSTDCXX_CXX
                                              VARIABLES CMAKE_CXX_FLAGS
        )
    endif()
endif()

if(ROCPROFSYS_BUILD_STATIC_LIBGCC)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_restore_variables(STATIC_LIBGCC_C VARIABLES CMAKE_C_FLAGS)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        rocprofiler_systems_restore_variables(STATIC_LIBGCC_CXX VARIABLES CMAKE_CXX_FLAGS)
    endif()
endif()

rocprofiler_systems_add_feature(CMAKE_C_FLAGS "C compiler flags")
rocprofiler_systems_add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")

# ----------------------------------------------------------------------------------------#
#
# Python
#
# ----------------------------------------------------------------------------------------#

if(ROCPROFSYS_USE_PYTHON)
    if(ROCPROFSYS_USE_PYTHON AND NOT ROCPROFSYS_BUILD_PYTHON)
        find_package(pybind11 REQUIRED)
    endif()

    include(ConfigPython)
    include(PyBind11Tools)

    rocprofiler_systems_watch_for_change(ROCPROFSYS_PYTHON_ROOT_DIRS _PYTHON_DIRS_CHANGED)

    if(_PYTHON_DIRS_CHANGED)
        unset(ROCPROFSYS_PYTHON_VERSION CACHE)
        unset(ROCPROFSYS_PYTHON_VERSIONS CACHE)
        unset(ROCPROFSYS_INSTALL_PYTHONDIR CACHE)
    else()
        foreach(_VAR PREFIX ENVS)
            rocprofiler_systems_watch_for_change(ROCPROFSYS_PYTHON_${_VAR} _CHANGED)

            if(_CHANGED)
                unset(ROCPROFSYS_PYTHON_ROOT_DIRS CACHE)
                unset(ROCPROFSYS_PYTHON_VERSIONS CACHE)
                unset(ROCPROFSYS_INSTALL_PYTHONDIR CACHE)
                break()
            endif()
        endforeach()
    endif()

    if(ROCPROFSYS_PYTHON_PREFIX AND ROCPROFSYS_PYTHON_ENVS)
        rocprofiler_systems_directory(
            FAIL
            PREFIX ${ROCPROFSYS_PYTHON_PREFIX}
            PATHS ${ROCPROFSYS_PYTHON_ENVS}
            OUTPUT_VARIABLE _PYTHON_ROOT_DIRS
        )
        set(ROCPROFSYS_PYTHON_ROOT_DIRS
            "${_PYTHON_ROOT_DIRS}"
            CACHE INTERNAL
            "Root directories for python"
        )
    endif()

    if(NOT ROCPROFSYS_PYTHON_VERSIONS AND ROCPROFSYS_PYTHON_VERSION)
        set(ROCPROFSYS_PYTHON_VERSIONS "${ROCPROFSYS_PYTHON_VERSION}")

        if(NOT ROCPROFSYS_PYTHON_ROOT_DIRS)
            rocprofiler_systems_find_python(_PY VERSION ${ROCPROFSYS_PYTHON_VERSION})
            set(ROCPROFSYS_PYTHON_ROOT_DIRS "${_PY_ROOT_DIR}" CACHE INTERNAL "" FORCE)
        endif()

        unset(ROCPROFSYS_PYTHON_VERSION CACHE)
        unset(ROCPROFSYS_INSTALL_PYTHONDIR CACHE)
    elseif(
        NOT ROCPROFSYS_PYTHON_VERSIONS
        AND NOT ROCPROFSYS_PYTHON_VERSION
        AND ROCPROFSYS_PYTHON_ROOT_DIRS
    )
        set(_PY_VERSIONS)

        foreach(_DIR ${ROCPROFSYS_PYTHON_ROOT_DIRS})
            rocprofiler_systems_find_python(_PY ROOT_DIR ${_DIR})

            if(NOT _PY_FOUND)
                continue()
            endif()

            if(NOT "${_PY_VERSION}" IN_LIST _PY_VERSIONS)
                list(APPEND _PY_VERSIONS "${_PY_VERSION}")
            endif()
        endforeach()

        set(ROCPROFSYS_PYTHON_VERSIONS "${_PY_VERSIONS}" CACHE INTERNAL "" FORCE)
    elseif(
        NOT ROCPROFSYS_PYTHON_VERSIONS
        AND NOT ROCPROFSYS_PYTHON_VERSION
        AND NOT ROCPROFSYS_PYTHON_ROOT_DIRS
    )
        rocprofiler_systems_find_python(_PY REQUIRED)
        set(ROCPROFSYS_PYTHON_ROOT_DIRS "${_PY_ROOT_DIR}" CACHE INTERNAL "" FORCE)
        set(ROCPROFSYS_PYTHON_VERSIONS "${_PY_VERSION}" CACHE INTERNAL "" FORCE)
    endif()

    rocprofiler_systems_watch_for_change(ROCPROFSYS_PYTHON_ROOT_DIRS)
    rocprofiler_systems_watch_for_change(ROCPROFSYS_PYTHON_VERSIONS)

    rocprofiler_systems_check_python_dirs_and_versions(FAIL)

    list(LENGTH ROCPROFSYS_PYTHON_VERSIONS _NUM_PYTHON_VERSIONS)

    if(_NUM_PYTHON_VERSIONS GREATER 1)
        set(ROCPROFSYS_INSTALL_PYTHONDIR
            "${CMAKE_INSTALL_LIBDIR}/python/site-packages"
            CACHE STRING
            "Installation prefix for python"
        )
    else()
        set(ROCPROFSYS_INSTALL_PYTHONDIR
            "${CMAKE_INSTALL_LIBDIR}/python${ROCPROFSYS_PYTHON_VERSIONS}/site-packages"
            CACHE STRING
            "Installation prefix for python"
        )
    endif()
else()
    set(ROCPROFSYS_INSTALL_PYTHONDIR
        "${CMAKE_INSTALL_LIBDIR}/python/site-packages"
        CACHE STRING
        "Installation prefix for python"
    )
endif()

rocprofiler_systems_watch_for_change(ROCPROFSYS_INSTALL_PYTHONDIR)
set(CMAKE_INSTALL_PYTHONDIR ${ROCPROFSYS_INSTALL_PYTHONDIR})

# ----------------------------------------------------------------------------------------#
#
# Compile definitions
#
# ----------------------------------------------------------------------------------------#

if("${CMAKE_BUILD_TYPE}" MATCHES "Release" AND NOT ROCPROFSYS_BUILD_DEBUG)
    add_target_flag_if_avail(rocprofiler-systems-compile-options "-g1")
endif()

target_compile_definitions(
    rocprofiler-systems-compile-definitions
    INTERFACE ROCPROFSYS_MAX_THREADS=${ROCPROFSYS_MAX_THREADS}
)

foreach(_LIB ${ROCPROFSYS_EXTENSION_LIBRARIES})
    get_target_property(_COMPILE_DEFS ${_LIB} INTERFACE_COMPILE_DEFINITIONS)
    if(_COMPILE_DEFS)
        foreach(_DEF ${_COMPILE_DEFS})
            if("${_DEF}" MATCHES "ROCPROFSYS_")
                target_compile_definitions(
                    rocprofiler-systems-compile-definitions
                    INTERFACE ${_DEF}
                )
            endif()
        endforeach()
    endif()
endforeach()
