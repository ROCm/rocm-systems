cmake_minimum_required ( VERSION 3.5.0 )

## Verbose output.
set ( CMAKE_VERBOSE_MAKEFILE on )

## Determine external build folder.
if( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    set( BUILD_FOLDER "lnx64a/B_rel" )
else()
    set( BUILD_FOLDER "lnx64a/B_dbg" )
endif()

## Check that the libhsail include and library directories are defined.
if ( NOT DEFINED LIBHSAIL_BUILD_FOLDER )
    set ( LIBHSAIL_BUILD_FOLDER ${BUILD_FOLDER} )
endif()

## Set ext runtime module name and project name.
set ( FINALIZE_NAME "hsa-ext-finalize" )
set ( FINALIZE_TARGET "${FINALIZE_NAME}64" )
project ( ${FINALIZE_TARGET} )

## Include the cmake_modules utils.cmake
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules" )
include ( utils )

## Compiler preproc definitions.
#add_definitions ( -D__linux__ )
add_definitions ( -DUNIX_OS )
add_definitions ( -DLINUX )
add_definitions ( -D__AMD64__ )
add_definitions ( -D__x86_64__ )
add_definitions ( -DAMD_INTERNAL_BUILD )
add_definitions ( -DLITTLEENDIAN_CPU=1 )
add_definitions ( -D HSA_DEPRECATED= )

## Get the package version. The defaults to 1.0.0.
get_version ( "1.0.0" )

set ( BUILD_VERSION_MAJOR ${VERSION_MAJOR} )
set ( BUILD_VERSION_MINOR ${VERSION_MINOR} )
set ( BUILD_VERSION_PATCH ${VERSION_PATCH} )
set ( LIB_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )
if ( DEFINED VERSION_BUILD )
    set ( BUILD_VERSION_PATCH "${BUILD_VERSION_PATCH}-${VERSION_BUILD}" )
endif ()
set ( BUILD_VERSION_STRING "${BUILD_VERSION_MAJOR}.${BUILD_VERSION_MINOR}.${BUILD_VERSION_PATCH}" )

## Find the hsakmt library and include files
find_file ( HSAKMT_INC NAMES "hsakmt.h" "libhsakmt/hsakmt.h" )
find_library ( HSAKMT_LIB libhsakmt.so )
get_filename_component ( HSAKMT_LIB_PATH ${HSAKMT_LIB} DIRECTORY )
get_filename_component ( HSAKMT_INC_PATH ${HSAKMT_INC} DIRECTORY )
include_directories ( ${HSAKMT_INC_PATH} )
link_directories (${HSAKMT_LIB_PATH})

## Find the hsa-runtime and include files
find_file ( HSA_INC "hsa/hsa.h" )
find_library ( HSA_LIB libhsa-runtime64.so )
get_filename_component ( HSA_LIB_PATH ${HSA_LIB} DIRECTORY )
get_filename_component ( HSA_INC_PATH ${HSA_INC} DIRECTORY )
include_directories ( ${HSA_INC_PATH} )
link_directories (${HSA_LIB_PATH})

## Find the external library files and set the link command
find_library ( HSAIL_LIB libhsail.a )
set ( LIBHSAIL -Wl,--no-whole-archive PRIVATE ${HSAIL_LIB} )

find_library ( HSAIL_AMD_LIB libhsail-amd.a )
set ( LIBHSAILAMD -Wl,--no-whole-archive PRIVATE ${HSAIL_AMD_LIB} )

find_library ( AMDHSAFIN_LIB amdhsafin64.a )
set ( LIBAMDHSAFIN -Wl,--no-whole-archive ${AMDHSAFIN_LIB} )

find_library ( LIBAMDHSACODE libamdhsacode.a )

find_library ( LIBCACHING libcaching.a )

find_library ( LIBSCP3 scSP3_R1000.a )

find_library ( LIBDWARF libdwarf.a )

find_library ( LIBELF libelf.a )

## External dependencies and directories
if ( NOT DEFINED REG_INCLUDE )
  set(REG_INCLUDE ${HSA_CLOSED_SOURCE_DIR}/drivers/inc/asic_reg)
endif()

if ( NOT EXISTS ${REG_INCLUDE}/si_id.h )
    MESSAGE ( FATAL_ERROR "Environment variable REG_INCLUDE is not set appropriately. REG_INCLUDE=${REG_INCLUDE}" )
else ()
    set ( REG_INCLUDE ${REG_INCLUDE} )
endif ()

if( NOT DEFINED EXT_SOURCE_DIR )
  set ( EXT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
endif()

if( NOT DEFINED OPEN_SOURCE_DIR )
  set ( OPEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." )
endif()

## Check for _GNU_SOURCE pthread extensions
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
CHECK_SYMBOL_EXISTS ( "pthread_attr_setaffinity_np" "pthread.h" HAVE_PTHREAD_ATTR_SETAFFINITY_NP )
CHECK_SYMBOL_EXISTS ( "pthread_rwlockattr_setkind_np" "pthread.h" HAVE_PTHREAD_RWLOCKATTR_SETKIND_NP )
unset(CMAKE_REQUIRED_DEFINITIONS)
if ( HAVE_PTHREAD_ATTR_SETAFFINITY_NP )
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HAVE_PTHREAD_ATTR_SETAFFINITY_NP )
endif()
if ( HAVE_PTHREAD_RWLOCKATTR_SETKIND_NP )
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HAVE_PTHREAD_RWLOCKATTR_SETKIND_NP )
endif()

## ------------------------- Linux Compiler and Linker options -------------------------
set ( CMAKE_CXX_FLAGS "-std=c++11 " )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fexceptions -fno-rtti -fvisibility=hidden -Wno-error=sign-compare -Wno-sign-compare -Wno-write-strings -Wno-deprecated-declarations -Wno-conversion-null -fno-math-errno -fno-threadsafe-statics -fmerge-all-constants -fms-extensions -Wno-error=comment -Wno-comment -Wno-error=pointer-arith -Wno-pointer-arith -fPIC" )

if ( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
    set  ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64  -msse -msse2" )
elseif ( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32" )
endif ()

if ( "${CMAKE_BUILD_TYPE}" STREQUAL Debug )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb" )
endif ()

set ( DRVDEF "${EXT_SOURCE_DIR}/make/finalize.so.def" )

set ( CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bdynamic -Wl,-z,noexecstack -Wl,--version-script=${DRVDEF}" )

set ( CMAKE_SKIP_BUILD_RPATH TRUE )

## Library path(s).
include_directories ( ${REG_INCLUDE} )
include_directories ( ${EXT_SOURCE_DIR}/.. )
include_directories ( ${EXT_SOURCE_DIR}/inc )
include_directories ( ${OPEN_SOURCE_DIR}/hsa-runtime )
include_directories ( ${OPEN_SOURCE_DIR}/hsa-runtime/inc )
include_directories ( ${OPEN_SOURCE_DIR}/hsa-runtime/core/inc )
include_directories ( ${OPEN_SOURCE_DIR}/hsa-runtime/core/common )
include_directories ( ${HSA_CLOSED_SOURCE_DIR}/drivers/hsa/compiler/ext_finalize/amdhsafin/Interface )
include_directories ( ${HSA_CLOSED_SOURCE_DIR}/drivers/hsa/compiler/finalizer/HSAIL/include )
include_directories ( ${HSA_CLOSED_SOURCE_DIR}/drivers/hsa/compiler/finalizer/HSAIL/hsail-tools/libHSAIL-AMD )
include_directories ( ${HSA_CLOSED_SOURCE_DIR}/drivers/hsa/compiler/finalizer/HSAIL/hsail-tools/libHSAIL )
include_directories ( ${HSA_CLOSED_SOURCE_DIR}/drivers/hsa/compiler/finalizer/HSAIL/hsail-tools/libHSAIL/build/${LIBHSAIL_BUILD_FOLDER} )

set ( FINALIZE_SRCS ${EXT_SOURCE_DIR}/finalize/hsa_ext_finalize.cpp
                    ${EXT_SOURCE_DIR}/finalize/program_context.cpp
                    ${EXT_SOURCE_DIR}/finalize/finalizer_manager.cpp
                    ${EXT_SOURCE_DIR}/runtime/amd_ext.cpp
                    ${EXT_SOURCE_DIR}/runtime/device_info.cpp
                    ${OPEN_SOURCE_DIR}/hsa-runtime/core/common/hsa_table_interface.cpp
                    ${OPEN_SOURCE_DIR}/hsa-runtime/core/common/shared.cpp
                    ${OPEN_SOURCE_DIR}/hsa-runtime/core/util/lnx/os_linux.cpp
)

add_library ( ${FINALIZE_TARGET} SHARED ${FINALIZE_SRCS} )

## Set the VERSION and SOVERSION values
set_property ( TARGET ${FINALIZE_TARGET} PROPERTY VERSION "${LIB_VERSION_STRING}" )
set_property ( TARGET ${FINALIZE_TARGET} PROPERTY SOVERSION "${BUILD_VERSION_MAJOR}" )

## Add the core runtime in the link
target_link_libraries (
    ${FINALIZE_TARGET}
    PRIVATE ${LIBAMDHSAFIN}
    PRIVATE ${LIBAMDHSACODE}
    PRIVATE ${LIBCACHING}
    PRIVATE ${LIBSCP3}
    PRIVATE ${LIBHSAILAMD}
    PRIVATE ${LIBHSAIL}
    PRIVATE ${LIBDWARF}
    PRIVATE ${LIBELF}
    PRIVATE hsa-runtime64
    PRIVATE hsakmt
    c stdc++ dl pthread rt
)

## If the build is Release, strip the target library
if ( "${CMAKE_BUILD_TYPE}" STREQUAL Release )
    add_custom_command ( TARGET ${FINALIZE_TARGET} POST_BUILD COMMAND ${CMAKE_STRIP} *.so )
endif ()

## Set install information
install ( TARGETS ${FINALIZE_TARGET} LIBRARY DESTINATION hsa/lib )
