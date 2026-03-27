## Compiler Preprocessor definitions.
add_definitions ( -DAMD_INTERNAL_BUILD )
add_definitions ( -DHSA_LARGE_MODEL= )
add_definitions ( -DHSA_DEPRECATED= )
add_definitions ( -DLITTLEENDIAN_CPU=1 )
# linux/registers/*_ip_offset.h headers use __maybe_unused as a struct attribute.
# Define it project-wide so it is never treated as a variable name regardless of platform.
if ( WIN32 )
  add_definitions ( "-DNOMINMAX" )
  add_definitions ( "-D__maybe_unused=" )
  add_definitions ( "-D__unused__=" )
else ()
  add_definitions ( "-D__maybe_unused=__attribute__((__unused__))" )
endif ()

## Compiler options
# Designated initializers in aql_profile.cpp require C++20 on MSVC
if ( MSVC )
  set(CMAKE_CXX_STANDARD 20)
  if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    add_compile_options("/MP$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
  endif()
else()
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ( NOT WIN32 )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()

add_definitions(-DNEW_TRACE_API=1)

## CLANG options
if ( NOT WIN32 AND "$ENV{CXX}" STREQUAL "/usr/bin/clang++" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1000000" )
endif()

## Enable debug trace
if ( DEFINED ENV{CMAKE_DEBUG_TRACE} )
  add_definitions ( -DDEBUG_TRACE=1 )
endif()

## Enable direct loading of AQL-profile HSA extension
if ( DEFINED ENV{CMAKE_LD_AQLPROFILE} )
  add_definitions (-DROCP_LD_AQLPROFILE=1)
endif()

## Build type
if ( NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "" )
  if ( DEFINED ENV{CMAKE_BUILD_TYPE} )
    set ( CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE} )
  endif()
endif()

## Installation prefix path
if ( NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CMAKE_PREFIX_PATH} )
  set ( CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )
endif()
set ( ENV{CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH} )

## Extend Compiler flags based on build type
string ( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
if ( "${CMAKE_BUILD_TYPE}" STREQUAL debug )
  if ( NOT MSVC )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb" )
  endif ()
  set ( CMAKE_BUILD_TYPE "debug" )
else ()
  set ( CMAKE_BUILD_TYPE "release" )
endif ()

## Extend Compiler flags based on Processor architecture
if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" )
  set ( NBIT 64 )
  set ( NBITSTR "64" )
  if ( NOT WIN32 )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64  -msse -msse2" )
  endif ()
elseif ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" )
  set ( NBIT 32 )
  set ( NBITSTR "" )
  if ( NOT WIN32 )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32" )
  endif ()
endif ()

## Find hsa-runtime
if ( WIN32 )
  # On Windows hsa-runtime64 package is not available.
  # Point AQLPROFILE_HSA_INCLUDE_DIR at the rocr-runtime inc/ folder so that
  # #include <hsa.h> (no hsa/ prefix on Windows) resolves directly.
  if ( NOT DEFINED AQLPROFILE_HSA_INCLUDE_DIR )
    # Default: rocr-runtime sibling project within the super-repo
    # CMAKE_SOURCE_DIR = projects/aqlprofile  ->  ../../ = super-repo root
    get_filename_component(_REPO_ROOT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)
    set ( AQLPROFILE_HSA_INCLUDE_DIR
          "${_REPO_ROOT}/projects/rocr-runtime/runtime/hsa-runtime/inc"
          CACHE PATH "Path to HSA runtime inc/ directory (rocr-runtime)" )
  endif ()
  message ( "---AQLPROFILE_HSA_INCLUDE_DIR: ${AQLPROFILE_HSA_INCLUDE_DIR}" )
  include_directories ( "${AQLPROFILE_HSA_INCLUDE_DIR}" )
else ()
  find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
endif ()

## Basic Tool Chain Information
message ( "----------------NBIT: ${NBIT}" )
message ( "-----------BuildType: ${CMAKE_BUILD_TYPE}" )
message ( "------------Compiler: ${CMAKE_CXX_COMPILER}" )
message ( "----Compiler-Version: ${CMAKE_CXX_COMPILER_VERSION}" )
message ( "------------API-path: ${API_PATH}" )
message ( "-----CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}" )
message ( "---CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" )
message ( "-CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}" )
message ( "-CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}" )
message ( "---------GPU_TARGETS: ${GPU_TARGETS}" )
