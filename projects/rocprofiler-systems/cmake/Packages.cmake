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
    "Provides flags and libraries for Dyninst (dynamic instrumentation")

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
#                               Dyninst
#
#----------------------------------------------------------------------------------------#

find_package(Dyninst ${hosttrace_FIND_QUIETLY} REQUIRED)
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

find_path(TBB_INCLUDE_DIR
    NAMES tbb/tbb.h
PATH_SUFFIXES include)

if(TBB_INCLUDE_DIR)
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
foreach(_TARG Dyninst::dyninst Boost::headers Boost::atomic
        Boost::system Boost::thread Boost::date_time)
    if(TARGET ${_TARG})
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

set(perfetto_DIR ${PROJECT_BINARY_DIR}/stuff/hosttrace-dyninst/perfetto)
if(NOT EXISTS "${perfetto_DIR}/.git")
    find_package(Git REQUIRED)
    execute_process(COMMAND
        ${GIT_EXECUTABLE} clone -b v17.0 https://android.googlesource.com/platform/external/perfetto/ ${perfetto_DIR})
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
