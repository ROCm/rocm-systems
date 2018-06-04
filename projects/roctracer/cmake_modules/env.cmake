## Build is not supported on Windows plaform
if ( WIN32 )
  message ( FATAL_ERROR "Windows build is not supported." )
endif ()

## Compiler Preprocessor definitions.
add_definitions ( -D__linux__ )
add_definitions ( -DUNIX_OS )
add_definitions ( -DLINUX )
add_definitions ( -D__AMD64__ )
add_definitions ( -D__x86_64__ )
add_definitions ( -DAMD_INTERNAL_BUILD )
add_definitions ( -DLITTLEENDIAN_CPU=1 )
add_definitions ( -DHSA_LARGE_MODEL= )
add_definitions ( -DHSA_DEPRECATED= )
add_definitions ( -D__HIP_PLATFORM_HCC__ )

## Linux Compiler options
set ( CMAKE_CXX_FLAGS "-std=c++11")
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants" )
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )

set ( CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bdynamic -Wl,-z,noexecstack" )

set ( CMAKE_SKIP_BUILD_RPATH TRUE )

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
  add_definitions ( -DROCP_LD_AQLPROFILE=1 )
endif()

## Make env vars
if ( NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "" )
  if ( DEFINED ENV{CMAKE_BUILD_TYPE} )
    set ( CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE} )
  endif()
endif()
if ( NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CMAKE_PREFIX_PATH} )
  set ( CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} )
endif()

set ( HIP_INC_DIR "$ENV{HIP_PATH}/include" )

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

## Find hsa-runtime headers/lib
find_file ( HSA_RUNTIME_INC "hsa.h" )
if ( "${HSA_RUNTIME_INC_PATH}" STREQUAL "" )
  find_file ( HSA_RUNTIME_INC "hsa/hsa.h" )
endif()
find_library ( HSA_RUNTIME_LIB "libhsa-runtime${NBIT}.so" )
get_filename_component ( HSA_RUNTIME_INC_PATH ${HSA_RUNTIME_INC} DIRECTORY )
get_filename_component ( HSA_RUNTIME_LIB_PATH ${HSA_RUNTIME_LIB} DIRECTORY )

find_library ( HSA_KMT_LIB "libhsakmt.so" )
get_filename_component ( HSA_KMT_LIB_PATH ${HSA_KMT_LIB} DIRECTORY )

set ( API_PATH ${HSA_RUNTIME_INC_PATH} )

## Basic Tool Chain Information
message ( "----------------NBIT: ${NBIT}" )
message ( "-----------BuildType: ${CMAKE_BUILD_TYPE}" )
message ( "------------Compiler: ${CMAKE_CXX_COMPILER}" )
message ( "----Compiler-Version: ${CMAKE_CXX_COMPILER_VERSION}" )
message ( "-----HSA-Runtime-Inc: ${HSA_RUNTIME_INC_PATH}" )
message ( "-----HSA-Runtime-Lib: ${HSA_RUNTIME_LIB_PATH}" )
message ( "------------API-path: ${API_PATH}" )
message ( "-----CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}" )
message ( "---CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" )
message ( "---------HIP_INC_DIR: ${HIP_INC_DIR}" )
