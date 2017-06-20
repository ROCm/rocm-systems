## Build is not supported on Windows plaform
if ( WIN32 )
  message ( FATAL_ERROR "Windows build is not supported." )
endif ()

## Compiler Preprocessor definitions.
add_definitions ( -DAMD_INTERNAL_BUILD )
add_definitions ( -DHSA_LARGE_MODEL= )
add_definitions ( -DHSA_DEPRECATED= )
add_definitions ( -DLITTLEENDIAN_CPU=1 )

## Linux Compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DNEW_TRACE_API=1)

## CLANG options
if ( "$ENV{CXX}" STREQUAL "/usr/bin/clang++" )
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
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb" )
  set ( CMAKE_BUILD_TYPE "debug" )
else ()
  set ( CMAKE_BUILD_TYPE "release" )
endif ()

## Extend Compiler flags based on Processor architecture
if ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" )
  set ( NBIT 64 )
  set ( NBITSTR "64" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64  -msse -msse2" )
elseif ( ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" )
  set ( NBIT 32 )
  set ( NBITSTR "" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32" )
endif ()

## Find hsa-runtime
find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

# find KFD thunk
find_package(hsakmt REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

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
