cmake_minimum_required(VERSION 2.8.0)

#
#  Setup build environment
#
#  1) Setup env var ROCR_INC_DIR and ROCR_LIB_DIR to point to
#     ROC Runtime header and libraries seperately
#
#     export ROCR_INC_DIR="Path to ROC Runtime header"
#
#     export ROCR_LIB_DIR="Path to ROC Runtime libraries"
#
#     export ROCT_LIB_DIR="Path to ROC Thunk libraries"
#
#  2) Make an new folder called build under root folder
#
#     mkdir build
#
#  3) Enter into folder of build, and run CMAKE to generate makefile
#     and make it
#
#     cd build; cmake ..; make
#

if(WIN32)
    message("Windows platfomr is not supported")
    rreturn()
endif()

if(NOT EXISTS $ENV{ROCR_INC_DIR}/hsa/hsa.h)
    message("ERROR: ROC Runtime headers can't be found under specified path")
    return()
endif()

#
# Flag to enable / disable verbose output.
#
set(CMAKE_VERBOSE_MAKEFILE on)

#
# Set core runtime module name
#
set(ROC_THUNK_NAME "hsakmt")
set(ROC_THUNK_LIBRARY "lib${ROC_THUNK_NAME}")
set(CORE_RUNTIME_NAME "hsa-runtime")
set(CORE_RUNTIME_TARGET "${CORE_RUNTIME_NAME}64")
set(CORE_RUNTIME_LIBRARY "lib${CORE_RUNTIME_TARGET}")

if(NOT EXISTS $ENV{ROCR_LIB_DIR}/${CORE_RUNTIME_LIBRARY}.so)
    message("ERROR: ROC Runtime libraries can't be found under sprcified path")
    return()
endif()

if(NOT EXISTS $ENV{ROCT_LIB_DIR}/${ROC_THUNK_LIBRARY}.so)
    message("ERROR: ROC Thunk libraries can't be found under sprcified path")
    return()
endif()

set(PROJECT_NAME "rocm_async")
set(TEST_NAME "${PROJECT_NAME}")
project(${PROJECT_NAME})

string(TOLOWER "${CMAKE_BUILD_TYPE}" tmp)
if("${tmp}" STREQUAL "debug")
    set(ISDEBUG "1")
    add_definitions(-DDEBUG)
endif()

if(ISDEBUG)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
else()
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
endif()

#
# Set the remaining compiler flags
#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")

include_directories($ENV{ROCR_INC_DIR})

link_directories($ENV{ROCR_LIB_DIR})

# Add sources that belong to the project
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} Src)

add_executable(rocm_async ${Src})
target_link_libraries(rocm_async hsa-runtime64)
