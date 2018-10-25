#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 2.8.0)

#
#   Required Defines on cmake command line
#
#   1) Set location of RSMI header files
#
#      ROCM_DIR="Root for ROCM install"
#
#   2) Set RSMITST_BLD_TYPE to either "Debug" or "Release".
#      If not set, the default value is "Debug" is bound.
#
#      RSMITST_BLD_TYPE=Debug or RSMITST_BLD_TYPE=Release
#
#   3) Set RSMITST_BLD_BITS to either "32" or "64"
#      If not set, the default value of "64" is bound.
#
#       RSMITST_BLD_BITS=32 or RSMITST_BLD_BITS=64
#
#   Building rocrtst Suite
#
#
#   1) Create build folder e.g. "rocrtst/build" - any name will do
#   2) Cd into build folder
#   3) Run "cmake .."
#   4) Run "make"
#

#
# Currently support for Windows platform is not present
#

#############################
# COMMON AREA
#############################
if(WIN32)
  message("rsmi library test suite is not supported on Windows platform")
  return()
endif()

#
# Process input variables
#

# Required Defines first:

set(RSMI_INC_DIR ${ROCM_DIR}/include)
set(RSMI_LIB_DIR ${ROCM_DIR}/lib)
#
# Determine RSMI Header files are present
#  (no external source dependencies)

# Determine RSMI Library file is present
#
if("${RSMITST_BLD_BITS}" STREQUAL 32)
  set (ONLY64STR "")
  set (IS64BIT 0)
else()
  set (ONLY64STR "64")
  set (IS64BIT 1)
endif()
#
if (${IS64BIT} EQUAL 0)
  if(NOT EXISTS ${RSMI_LIB_DIR}/librocm_smi32.so)
    message("ERROR: ${RSMI_LIB_DIR}/librocm_smi32.so doesn't exist. Check value of ROCM_DIR define")
    return()
  endif()
else()
  if(NOT EXISTS ${RSMI_LIB_DIR}/librocm_smi64.so)
    message("ERROR: Define RSMI_LIB_DIR pointing to RSMI library is not set")
    message(" missing: ${RSMI_LIB_DIR}/librocm_smi.so")
    return()
  endif()
endif()

string(TOLOWER "${RSMITST_BLD_TYPE}" tmp)
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
message("-------------IS64BIT: " ${IS64BIT})
message("-----------BuildType: " ${BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
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
# Extend the compiler flags for 64-bit builds
#
if (IS64BIT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64  -msse -msse2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
endif()

#
# Add compiler flags to include symbol information for debug builds
#
if(ISDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
endif()
MESSAGE("ISDEBUG STEP:Done")


set(RSMITST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Set Name for Google Test Framework and build it as a
# static library to be linked with user test programs
#
set(GOOGLE_TEST_FRWK_NAME "google-test-frwk${ONLY64STR}")
add_subdirectory(${RSMITST_ROOT}/gtest "${PROJECT_BINARY_DIR}/gtest")
set (RSMITST_LIBS ${RSMITST_LIBS} ${GOOGLE_TEST_FRWK_NAME})

#
#
# Other source directories
aux_source_directory(${RSMITST_ROOT}/functional functionalSources)

#
# Specify the directory containing various libraries of ROCR
# to be linked against for building ROC Perf applications
#
LINK_DIRECTORIES(${RSMI_LIB_DIR})

#
# Extend the list of libraries to be used for linking rsmi apps
#
if (IS64BIT)
  set(RSMITST_LIBS ${RSMITST_LIBS} rocm_smi64)
  set(RSMITST "rsmitst64")
else()
  set(RSMITST_LIBS ${RSMITST_LIBS} rocm_smi32)
  set(RSMITST "rsmitst")
endif()


#
# Source files for building rocrtst
#
aux_source_directory(${RSMITST_ROOT} rsmitstSources)

# Header file include path

include_directories(${RSMI_INC_DIR})
include_directories(${RSMITST_ROOT})
include_directories(${RSMITST_ROOT}/..)
include_directories(${RSMITST_ROOT}/gtest/include)

# Build rules
add_executable(${RSMITST} ${rsmitstSources} ${functionalSources})

#target_link_libraries(${ROCRTST} ${ROCRTST_LIBS} c stdc++ dl pthread rt numa hwloc)

target_link_libraries(${RSMITST} ${RSMITST_LIBS} c stdc++ pthread)

install(TARGETS ${RSMITST}
        ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/lib
        LIBRARY DESTINATION ${PROJECT_BINARY_DIR}/lib
        RUNTIME DESTINATION ${PROJECT_BINARY_DIR}/bin)

