# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#                       External Packages are found here
#
##########################################################################################

add_interface_library(hosttrace-headers
    "Provides minimal set of include flags to compile with hosttrace")
add_interface_library(hosttrace-threading
    "Enables multithreading support")
add_interface_library(hosttrace-dyninst
    "Provides flags and libraries for Dyninst (dynamic instrumentation)")
add_interface_library(hosttrace-roctracer
    "Provides flags and libraries for roctracer")

# include threading because of rooflines
target_link_libraries(hosttrace-headers INTERFACE hosttrace-threading)

#----------------------------------------------------------------------------------------#
#
#                               Threading
#
#----------------------------------------------------------------------------------------#

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
endif()

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)
find_package(Threads ${hosttrace_FIND_QUIETLY} ${hosttrace_FIND_REQUIREMENT})

if(Threads_FOUND)
    target_link_libraries(hosttrace-threading INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif()

if(pthread_LIBRARY AND NOT WIN32)
    target_link_libraries(hosttrace-threading INTERFACE ${pthread_LIBRARY})
endif()


#----------------------------------------------------------------------------------------#
#
#                               roctracer
#
#----------------------------------------------------------------------------------------#

if(HOSTTRACE_USE_ROCTRACER)
    find_package(roctracer ${hosttrace_FIND_QUIETLY} REQUIRED)
    find_package(hip ${hosttrace_FIND_QUIETLY} REQUIRED)
    target_compile_definitions(hosttrace-roctracer INTERFACE HOSTTRACE_USE_ROCTRACER)
    target_link_libraries(hosttrace-roctracer INTERFACE hip::host roctracer::roctracer)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${roctracer_LIBRARY_DIRS}")
endif()

#----------------------------------------------------------------------------------------#
#
#                               Dyninst
#
#----------------------------------------------------------------------------------------#

find_package(Dyninst ${hosttrace_FIND_QUIETLY} REQUIRED
    COMPONENTS dyninstAPI parseAPI instructionAPI symtabAPI)
set(_BOOST_COMPONENTS atomic system thread date_time)
set(hosttrace_BOOST_COMPONENTS "${_BOOST_COMPONENTS}" CACHE STRING
    "Boost components used by Dyninst in hosttrace")
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost QUIET REQUIRED
    COMPONENTS ${hosttrace_BOOST_COMPONENTS})

# some installs of dyninst don't set this properly
if(EXISTS "${DYNINST_INCLUDE_DIR}" AND NOT DYNINST_HEADER_DIR)
    get_filename_component(DYNINST_HEADER_DIR "${DYNINST_INCLUDE_DIR}" REALPATH CACHE)
else()
    find_path(DYNINST_HEADER_DIR
        NAMES BPatch.h dyninstAPI_RT.h
        HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
        PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
        PATH_SUFFIXES include)
endif()

# useful for defining the location of the runtime API
find_library(DYNINST_API_RT dyninstAPI_RT
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
    find_path(TBB_INCLUDE_DIR
        NAMES tbb/tbb.h
        PATH_SUFFIXES include)
endif()

if(TBB_INCLUDE_DIR AND NOT TBB_INCLUDE_DIRS)
    set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
endif()

if(DYNINST_API_RT)
    target_compile_definitions(hosttrace-dyninst INTERFACE
        DYNINST_API_RT="${DYNINST_API_RT}")
endif()

if(Boost_DIR)
    get_filename_component(Boost_RPATH_DIR "${Boost_DIR}" DIRECTORY)
    get_filename_component(Boost_RPATH_DIR "${Boost_RPATH_DIR}" DIRECTORY)
    if(EXISTS "${Boost_RPATH_DIR}" AND IS_DIRECTORY "${Boost_RPATH_DIR}")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${Boost_RPATH_DIR}")
    endif()
endif()

add_rpath(${DYNINST_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(hosttrace-dyninst INTERFACE
    ${DYNINST_LIBRARIES} ${Boost_LIBRARIES})
foreach(_TARG dyninst dyninstAPI instructionAPI symtabAPI parseAPI headers atomic system thread date_time TBB)
    if(TARGET Dyninst::${_TARG})
        target_link_libraries(hosttrace-dyninst INTERFACE Dyninst::${_TARG})
    elseif(TARGET Boost::${_TARG})
        target_link_libraries(hosttrace-dyninst INTERFACE Boost::${_TARG})
    elseif(TARGET ${_TARG})
        target_link_libraries(hosttrace-dyninst INTERFACE ${_TARG})
    endif()
endforeach()
target_include_directories(hosttrace-dyninst SYSTEM INTERFACE
    ${TBB_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${DYNINST_HEADER_DIR})
target_compile_definitions(hosttrace-dyninst INTERFACE hosttrace_USE_DYNINST)

if(DYNINST_API_RT)
    add_cmake_defines(DYNINST_API_RT VALUE QUOTE DEFAULT)
else()
    add_cmake_defines(DYNINST_API_RT VALUE QUOTE)
endif()

#----------------------------------------------------------------------------------------#
#
#                               Perfetto
#
#----------------------------------------------------------------------------------------#

set(perfetto_DIR ${PROJECT_SOURCE_DIR}/external/perfetto)
checkout_git_submodule(
    RELATIVE_PATH       external/perfetto
    WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
    REPO_URL            https://android.googlesource.com/platform/external/perfetto
    REPO_BRANCH         v17.0
    TEST_FILE           sdk/perfetto.cc)

#----------------------------------------------------------------------------------------#
#
#                               ELFIO
#
#----------------------------------------------------------------------------------------#

if(HOSTTRACE_BUILD_DEVICETRACE)
    checkout_git_submodule(
        RELATIVE_PATH       external/elfio
        WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
        REPO_URL            https://github.com/jrmadsen/ELFIO.git
        REPO_BRANCH         set-offset-support)

    add_subdirectory(external/elfio)
endif()

#----------------------------------------------------------------------------------------#
#
#                               Clang Tidy
#
#----------------------------------------------------------------------------------------#

# clang-tidy
macro(HOSTTRACE_ACTIVATE_CLANG_TIDY)
    if(HOSTTRACE_USE_CLANG_TIDY)
        find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
        add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
        if(NOT CLANG_TIDY_COMMAND)
            timemory_message(WARNING "HOSTTRACE_USE_CLANG_TIDY is ON but clang-tidy is not found!")
            set(HOSTTRACE_USE_CLANG_TIDY OFF)
        else()
            set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})

            # Create a preprocessor definition that depends on .clang-tidy content so
            # the compile command will change when .clang-tidy changes.  This ensures
            # that a subsequent build re-runs clang-tidy on all sources even if they
            # do not otherwise need to be recompiled.  Nothing actually uses this
            # definition.  We add it to targets on which we run clang-tidy just to
            # get the build dependency on the .clang-tidy file.
            file(SHA1 ${CMAKE_CURRENT_LIST_DIR}/.clang-tidy clang_tidy_sha1)
            set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
            unset(clang_tidy_sha1)
        endif()
    endif()
endmacro()

#------------------------------------------------------------------------------#
#
#                   clang-format target
#
#------------------------------------------------------------------------------#

find_program(HOSTTRACE_CLANG_FORMAT_EXE
    NAMES
        clang-format-12
        clang-format-11
        clang-format-10
        clang-format-9
        clang-format)

if(HOSTTRACE_CLANG_FORMAT_EXE)
    file(GLOB sources
        ${PROJECT_SOURCE_DIR}/src/*.cpp)
    file(GLOB headers
        ${PROJECT_SOURCE_DIR}/include/*.hpp)
    file(GLOB_RECURSE examples
        ${PROJECT_SOURCE_DIR}/examples/*.cpp
        ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    add_custom_target(format
        ${HOSTTRACE_CLANG_FORMAT_EXE} -i ${sources} ${headers} ${examples}
        COMMENT "Running ${HOSTTRACE_CLANG_FORMAT_EXE}...")
else()
    message(AUTHOR_WARNING "clang-format could not be found. format build target not available.")
endif()

#----------------------------------------------------------------------------------------#
#   configure submodule
#----------------------------------------------------------------------------------------#

set(TIMEMORY_INSTALL_HEADERS        OFF CACHE BOOL "Disable timemory header install")
set(TIMEMORY_INSTALL_CONFIG         OFF CACHE BOOL "Disable timemory cmake configuration install")
set(TIMEMORY_INSTALL_ALL            OFF CACHE BOOL "Disable install target depending on all target")
set(TIMEMORY_BUILD_C                OFF CACHE BOOL "Disable timemory C library")
set(TIMEMORY_BUILD_FORTRAN          OFF CACHE BOOL "Disable timemory Fortran library")
set(TIMEMORY_BUILD_TOOLS            OFF CACHE BOOL "Ensure timem executable is built")
set(TIMEMORY_BUILD_EXCLUDE_FROM_ALL ON  CACHE BOOL "Set timemory to only build dependencies")
set(TIMEMORY_QUIET_CONFIG           ON  CACHE BOOL "Make timemory configuration quieter")

# timemory feature settings
set(TIMEMORY_USE_GOTCHA             ON  CACHE BOOL "Enable GOTCHA support in timemory")
set(TIMEMORY_USE_PERFETTO           OFF CACHE BOOL "Disable perfetto support in timemory")
# timemory feature build settings
set(TIMEMORY_BUILD_GOTCHA           ON  CACHE BOOL "Enable building GOTCHA library from submodule")
# timemory build settings
set(TIMEMORY_TLS_MODEL "global-dynamic" CACHE STRING "Thread-local static model" FORCE)

checkout_git_submodule(
    RELATIVE_PATH       external/timemory
    WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
    REPO_URL            https://github.com/NERSC/timemory.git
    REPO_BRANCH         develop)

hosttrace_save_variables(BUILD_CONFIG
    BUILD_SHARED_LIBS
    BUILD_STATIC_LIBS
    CMAKE_POSITION_INDEPENDENT_CODE)

# ensure timemory builds PIC static libs so that we don't have to install timemory shared lib
set(BUILD_SHARED_LIBS ON)
set(BUILD_STATIC_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external/timemory)

hosttrace_restore_variables(BUILD_CONFIG
    BUILD_SHARED_LIBS
    BUILD_STATIC_LIBS
    CMAKE_POSITION_INDEPENDENT_CODE)
