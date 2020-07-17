#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 3.5.0)

#   Required Defines on cmake command line
#
#   1) Set location of ROCm root to pick up RDC and RDC headers and RDC lib
#
#      ROCM_DIR="Root for ROCM install"
#
#   2) Set location of RDC root to pick up RDC and RDC headers and RDC lib
#
#      RDC_DIR="Root for RDC install"
#
#   3) Set RDCTST_BLD_TYPE to either "Debug" or "Release".
#      If not set, the default value is "Debug" is bound.
#
#      RDCTST_BLD_TYPE=Debug or RDCTST_BLD_TYPE=Release
#
#   Building rocrtst Suite
#
#
#   1) Create build folder e.g. "rdctst/build" - any name will do
#   2) Cd into build folder
#   3) Run "cmake .."
#   4) Run "make"
#
# Currently support for Windows platform is not present

#############################
# COMMON AREA
#############################
if(WIN32)
  message("rdc library test suite is not supported on Windows platform")
  return()
endif()

#
# Process input variables
#

# Required Defines first:

set(RDC_CLIENT_INSTALL_PREFIX "opt/rocm")
set(RDC_INC_DIR ${ROCM_DIR}/${RDC_CLIENT_INSTALL_PREFIX}/include)
set(RDC_LIB_DIR ${ROCM_DIR}/${RDC_CLIENT_INSTALL_PREFIX}/rdc/lib)

#
# Determine RDC Header files are present
#  (no external source dependencies)


string(TOLOWER "${RDCTST_BLD_TYPE}" tmp)
if("${tmp}" STREQUAL release)
  set(BUILD_TYPE "Release")
  set(ISDEBUG 0)
else()
  set(BUILD_TYPE "Debug")
  set(ISDEBUG 1)
endif()

#
# Print out the build configuration being used:
#
#   Build Src directory
#   Build Binary directory
#   Build Type: Debug Vs Release, 32 Vs 64
#   Compiler Version, etc
#
message("")
message("Build Configuration:")
message("-----------BuildType: " ${BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------------ROCM_DIR: " ${ROCM_DIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------RDC Lib Dir: " ${RDC_LIB_DIR})
message("--------RDC Inc Dir: " ${RDC_INC_DIR})
message("")

#
# Set the build type based on user input
#
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
#
# Flag to enable / disable verbose output.
#
set(CMAKE_VERBOSE_MAKEFILE on)
#
# Compiler pre-processor definitions.
#
# Define MACRO "DEBUG" if build type is "Debug"
if(${BUILD_TYPE} STREQUAL "Debug")
add_definitions(-DDEBUG)
endif()

add_definitions(-D__linux__)
add_definitions(-DLITTLEENDIAN_CPU=1)

#
# Linux Compiler options
#
set(CMAKE_CXX_FLAGS "-std=c++11 ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

#
# Add compiler flags to include symbol information for debug builds
#
if(ISDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
endif()

set(RDCTST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(RDCTST "rdctst")
set(RDCTST_LIBS "rdc_bootstrap" )

# Set Name for Google Test Framework and build it as a
# static library to be linked with user test programs
#
set(GOOGLE_TEST_FRWK_NAME "gtest")
add_subdirectory(${RDCTST_ROOT}/gtest "${PROJECT_BINARY_DIR}/gtest")
set (RDCTST_LIBS ${RDCTST_LIBS} ${GOOGLE_TEST_FRWK_NAME})

set(RDCTST_LIBS ${RDCTST_LIBS} ${GOOGLE_TEST_FRWK_NAME})

# Other source directories
aux_source_directory(${RDCTST_ROOT}/functional functionalSources)

#
# Specify the directory containing various libraries of ROCR
# to be linked against for building ROC Perf applications
#

link_directories(${ROCM_DIR}/lib)
#
# Source files for building rocrtst
#
aux_source_directory(${RDCTST_ROOT} rdctstSources)

# Build rules
add_executable(${RDCTST} ${rdctstSources} ${functionalSources})

# Header file include path
target_include_directories(${RDCTST} PRIVATE ${RDC_INC_DIR}
                                     PRIVATE ${ROCM_DIR}/include
                                     PRIVATE ${RDCTST_ROOT}/..
                                     PRIVATE ${RDCTST_ROOT}/gtest/include)

target_link_libraries(${RDCTST} ${RDCTST_LIBS} c stdc++ pthread)

