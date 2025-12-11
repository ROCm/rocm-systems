#
#   Required Defines on cmake command line
#
#   1) Set location of ROCR header files
#
#      ROCM_DIR="Root for RocM install"
#
#   2) Set ROCRTST_BLD_TYPE to either "Debug" or "Release".
#      If not set, the default value is "Debug" is bound.
#
#      ROCRTST_BLD_TYPE=Debug or ROCRTST_BLD_TYPE=Release
#
#   3) Set ROCRTST_BLD_BITS to either "32" or "64"
#      If not set, the default value of "64" is bound.
#
#       ROCRTST_BLD_BITS=32 or ROCRTST_BLD_BITS=64
#
#   4) Set TARGET_DEVICES to indicate gpu types for kernel
#      builds (e.g., "gfx803;gfx900; ...")
#
#   Building rocrtst Suite
#
#
#   1) Create build folder e.g. "rocrtst/build" - any name will do
#   2) Cd into build folder
#   3) Run "cmake .."
#   4) Run "make"
#

cmake_minimum_required(VERSION 3.5.0)

# Set Name for Samples Project
#

set(PROJECT_NAME "rocrtst64")
project (${PROJECT_NAME})

if ( NOT DEFINED BUILD_SHARED_LIBS )
  set ( BUILD_SHARED_LIBS ON )
endif()

# For DEB/RPM generation
if(BUILD_SHARED_LIBS)
  set ( CPACK_PACKAGE_NAME "rocrtst" )
else()
  set ( CPACK_RPM_PACKAGE_NAME "rocrtst-static" )
  set ( CPACK_DEBIAN_PACKAGE_NAME "rocrtst-static" )
endif()

set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc." )
set ( CPACK_PACKAGE_DESCRIPTION "This package includes rocrtst and a convenience script to run the test suite" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Test suite for ROCr" )
set ( CPACK_PACKAGE_VERSION_MAJOR "1" )
set ( CPACK_PACKAGE_VERSION_MINOR "0" )
set ( CPACK_PACKAGE_VERSION_PATCH "0" )

# Make proper version for appending
# Default Value is 99999, setting it first
set(ROCM_VERSION_FOR_PACKAGE "99999")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
endif()
set (PACKAGE_VERSION_STR "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")
set ( CPACK_PACKAGE_VERSION "${PACKAGE_VERSION_STR}")

set ( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"  CACHE STRING "Default packaging prefix." )
set ( CPACK_GENERATOR "DEB;RPM"  CACHE STRING "Default packaging generators." )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCR-Runtime" )
set ( CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/RadeonOpenCompute/ROCR-Runtime" )
set ( CPACK_RPM_PACKAGE_AUTOREQ 0 )
set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )
## Debian package values
set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
if( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
endif()

if(BUILD_SHARED_LIBS)
  set (CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core,hsa-rocr")
else()
  set (CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core,hsa-rocr-static-dev")
endif()

## RPM package variables
set ( CPACK_RPM_PACKAGE_RELEASE "local" )
if( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
  set ( CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} )
endif()
## Add os details to rpm package name. For deb packages, its set from build environment
# Modify line 87 to be "if the env. variable DISTRO_NAME IS centos or rhel"
if (DEFINED ENV{DISTRO_NAME} AND
  (ENV{DISTRO_NAME} MATCHES "centos" OR
   ENV{DISTRO_NAME} MATCHES "rhel" OR
   ENV{DISTRO_NAME} MATCHES "sles"))
  execute_process(COMMAND rpm --eval %{?dist}
          RESULT_VARIABLE PROC_RESULT
          OUTPUT_VARIABLE EVAL_RESULT
          OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string (APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
  endif()
endif()


if(BUILD_SHARED_LIBS)
  set (CPACK_RPM_PACKAGE_REQUIRES "rocm-core,hsa-rocr")
else()
  set (CPACK_RPM_PACKAGE_REQUIRES "rocm-core,hsa-rocr-static-devel")
endif()

set(DEFAULT_TARGETS "gfx700;gfx701;gfx702;gfx801;gfx802;gfx803;gfx805;gfx810"
                    "gfx900;gfx902;gfx904;gfx906;gfx908;gfx909;gfx90a;gfx90c;gfx942;gfx950"
                    "gfx1010;gfx1011;gfx1012;gfx1013;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
                    "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")

#
# Currently support for Windows platform is not present
#

#############################
# COMMON AREA
#############################
if(WIN32)
  message("rocrtst Suite is not supported on Windows platform")
  return()
endif()

#
# Process input variables
#

# Required Defines first:
find_package(hsa-runtime64 REQUIRED)
find_package(amd_smi REQUIRED)

set (ONLY64STR "64")

# Find LLVM/Clang using find_package instead of hardcoded LLVM_DIR
find_package(LLVM REQUIRED CONFIG)
if(NOT LLVM_FOUND)
  message(FATAL_ERROR "LLVM not found. Cannot build kernels.")
endif()

# Add LLVM CMake modules to module path
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
include(AddLLVM)

# Find clang executable from LLVM package
find_program(CLANG
  NAMES clang
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "Found LLVM: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM Binary Dir: ${LLVM_TOOLS_BINARY_DIR}")
message(STATUS "LLVM Library Dir: ${LLVM_LIBRARY_DIR}")
message(STATUS "Found Clang: ${CLANG}")

# Set OpenCL version
if (DEFINED OPENCL_VER)
  set(OPENCL_VER ${OPENCL_VER})
else()
  message(STATUS "OPENCL_VER not defined, using default: 2.0")
  set(OPENCL_VER "2.0")
endif()

# Find OpenCL headers using clang's resource directory
execute_process(
  COMMAND ${CLANG} -print-resource-dir
  OUTPUT_VARIABLE CLANG_RESOURCE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE CLANG_RESULT
)

if(NOT CLANG_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get clang resource directory")
endif()

set(OPENCL_INC_DIR "${CLANG_RESOURCE_DIR}/include")

if(NOT EXISTS "${OPENCL_INC_DIR}/opencl-c.h")
  message(FATAL_ERROR "opencl-c.h not found in ${OPENCL_INC_DIR}")
endif()

message(STATUS "OpenCL headers: ${OPENCL_INC_DIR}")
message(STATUS "OpenCL version: ${OPENCL_VER}")

# Find AMD Device Libraries (bitcode files) using find_package or manual search
find_package(AMDDeviceLibs QUIET CONFIG)

if(AMDDeviceLibs_FOUND)
  # Use the package-provided location
  message(STATUS "Found AMDDeviceLibs via CONFIG")
  # Try to get the bitcode directory from any target
  if(TARGET AMDDeviceLibs::oclc_isa_version_803)
    get_target_property(DEVICE_LIBS_LOCATION AMDDeviceLibs::oclc_isa_version_803 IMPORTED_LOCATION)
    get_filename_component(BITCODE_DIR "${DEVICE_LIBS_LOCATION}" DIRECTORY)
  elseif(TARGET AMDDeviceLibs::oclc_isa_version_900)
    get_target_property(DEVICE_LIBS_LOCATION AMDDeviceLibs::oclc_isa_version_900 IMPORTED_LOCATION)
    get_filename_component(BITCODE_DIR "${DEVICE_LIBS_LOCATION}" DIRECTORY)
  else()
    # Fallback to manual search even if package was found
    set(AMDDeviceLibs_FOUND FALSE)
  endif()
endif()

if(NOT AMDDeviceLibs_FOUND)
  # Fallback: search relative to LLVM installation
  message(STATUS "Searching for AMD Device Libraries manually...")

  find_path(BITCODE_DIR 
    NAMES "opencl.bc" "opencl.amdgcn.bc" "oclc_isa_version_803.bc"
    HINTS
      "${LLVM_LIBRARY_DIR}/amdgcn/bitcode"
      "${LLVM_LIBRARY_DIR}/../amdgcn/bitcode"
      "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode"
      "${CMAKE_INSTALL_PREFIX}/lib/amdgcn/bitcode"
      "${CMAKE_PREFIX_PATH}/amdgcn/bitcode"
      "${CMAKE_PREFIX_PATH}/lib/amdgcn/bitcode"
    NO_DEFAULT_PATH
  )
  
  if(NOT BITCODE_DIR)
    message(FATAL_ERROR 
      "AMD Device Libraries (bitcode) not found!\n"
      "Searched locations:\n"
      "  ${LLVM_LIBRARY_DIR}/amdgcn/bitcode\n"
      "  ${LLVM_LIBRARY_DIR}/../amdgcn/bitcode\n"
      "  ${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode\n"
      "  ${CMAKE_INSTALL_PREFIX}/lib/amdgcn/bitcode\n"
      "Please ensure device-libs are built and installed.")
  endif()
endif()

message(STATUS "Found Device Libraries: ${BITCODE_DIR}")

# Validate bitcode files exist
if(NOT EXISTS "${BITCODE_DIR}")
  message(FATAL_ERROR "BITCODE_DIR does not exist: ${BITCODE_DIR}")
endif()

# Check for either old or new device-libs layout
if(EXISTS "${BITCODE_DIR}/opencl.amdgcn.bc")
  message(STATUS "Using new device-libs layout (opencl.amdgcn.bc)")
elseif(EXISTS "${BITCODE_DIR}/opencl.bc")
  message(STATUS "Using old device-libs layout (opencl.bc)")
else()
  message(WARNING 
    "No standard device library files found in ${BITCODE_DIR}\n"
    "Expected either 'opencl.amdgcn.bc' or 'opencl.bc'\n"
    "Will attempt to continue, but kernel builds may fail.")
endif()

# Target devices configuration
if (NOT DEFINED TARGET_DEVICES)
  message("No target devices provided on command line")
  message("  e.g., cmake -DTARGET_DEVICES=\"gfx803;gfx900;gfx...\" ..")
  message("  Using default target of ${DEFAULT_TARGETS}")
  list(APPEND TARGET_DEVICES ${DEFAULT_TARGETS})
endif()

# Build type configuration
string(TOLOWER "${ROCRTST_BLD_TYPE}" tmp)
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
message("------Target Devices: ${TARGET_DEVICES}")
message("----------Clang path: " ${CLANG})
message("------OpenCL Inc Dir: " ${OPENCL_INC_DIR})
message("-------OpenCL version " ${OPENCL_VER})
message("-------BITCODE_DIR: " ${BITCODE_DIR})
message("------------LLVM Dir: " ${LLVM_DIR})
message("")

set(KERNELS_DIR ${PROJECT_SOURCE_DIR}/kernels)
#
# Set the build type based on user input
#
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
#
# Compiler pre-processor definitions.
#
# Define MACRO "DEBUG" if build type is "Debug"
if(${BUILD_TYPE} STREQUAL "Debug")
add_definitions(-DDEBUG)
endif()

#add_definitions(-D__linux__)
add_definitions(-DLITTLEENDIAN_CPU=1)

#
# Linux Compiler options
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmerge-all-constants")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

#
# Add compiler flags to include symbol information for debug builds
#
if(ISDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
endif()
MESSAGE("ISDEBUG STEP:Done")

set(ROCRTST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Set Name for Google Test Framework and build it as a
# static library to be linked with user test programs
#
set(GOOGLE_TEST_FRWK_NAME "google-test-frwk${ONLY64STR}")
add_subdirectory(${ROCRTST_ROOT}/gtest "${PROJECT_BINARY_DIR}/gtest" EXCLUDE_FROM_ALL)
set (ROCRTST_LIBS ${ROCRTST_LIBS} ${GOOGLE_TEST_FRWK_NAME})

MESSAGE("ROCRTST_LIBS SET STEP:Done")
#
#
# Other source directories
aux_source_directory(${ROCRTST_ROOT}/common common_srcs)

#
# Extend the list of libraries to be used for linking ROC Perf Apps
#
set(ROCRTST_LIBS ${ROCRTST_LIBS} hsa-runtime64::hsa-runtime64)
set(ROCRTST_LIBS ${ROCRTST_LIBS} amd_smi)


# Set Name for rocrtst
MESSAGE(${ROCRTST_LIBS})
set(ROCRTST "rocrtst${ONLY64STR}")

#
# Source files for building rocrtst
#
aux_source_directory(${ROCRTST_ROOT}/suites/performance performanceSources)
aux_source_directory(${ROCRTST_ROOT}/suites/functional functionalSources)
aux_source_directory(${ROCRTST_ROOT}/suites/negative negativeSources)
aux_source_directory(${ROCRTST_ROOT}/suites/stress stressSources)
aux_source_directory(${ROCRTST_ROOT}/suites/test_common testCommonSources)

# Header file include path

include_directories(${ROCRTST_ROOT})
include_directories(${ROCRTST_ROOT}/gtest/include)
include_directories(${ROCRTST_ROOT}/thirdparty/include/)

# Custom command set for code objects.
set (HSACO_TARG_LIST "")

# Use this function to build any samples that have kernels to be built
function(build_kernel S_NAME TARG_DEV)
  set(KERNEL_DIR ${PROJECT_BINARY_DIR}/${TARG_DEV})
  set(SNAME_KERNEL "${S_NAME}_kernels.hsaco")

  set(TARG_NAME "${S_NAME}_hsaco.${TARG_DEV}")
  set(HSACO_TARG_LIST ${HSACO_TARG_LIST} "${KERNEL_DIR}/${SNAME_KERNEL}" PARENT_SCOPE)
  string(SUBSTRING ${TARG_DEV} 3 -1 gfxNum)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
   "-D ROCRTST_GPU=0x${gfxNum} -x cl -target amdgcn-amd-amdhsa -include ${OPENCL_INC_DIR}/opencl-c.h -mcpu=${TARG_DEV} ${BITCODE_ARGS} -cl-std=CL${OPENCL_VER} -mcode-object-version=4 ${CL_FILE_LIST} -o ${KERNEL_DIR}/${SNAME_KERNEL}")
  add_custom_command(OUTPUT "${KERNEL_DIR}/${SNAME_KERNEL}" COMMAND ${CLANG} ${CLANG_ARG_LIST} DEPENDS ${CL_FILE_LIST} ${CLANG} COMMENT "BUILDING ${KERNEL_DIR}/${SNAME_KERNEL}" VERBATIM)
endfunction(build_kernel)

function(build_sample_for_devices S_NAME)
  foreach(t ${TARGET_DEVICES})
    build_kernel(${S_NAME} ${t})
  endforeach(t)
  set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)
endfunction(build_sample_for_devices)

# Make directories for each possible target device
# List of symlinks per directory
set(ROCRTST_LINKS_LIST "")
foreach(td ${TARGET_DEVICES})
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/${td})
  add_custom_command(OUTPUT "${PROJECT_BINARY_DIR}/${td}/${ROCRTST}" COMMAND ${CMAKE_COMMAND} -E create_symlink "../${ROCRTST}" "${td}/${ROCRTST}" COMMENT "BUILDING ${td}/${ROCRTST}" VERBATIM)
  set(ROCRTST_LINKS_LIST ${ROCRTST_LINKS_LIST} "${PROJECT_BINARY_DIR}/${td}/${ROCRTST}")
  install ( DIRECTORY ${PROJECT_BINARY_DIR}/${td} DESTINATION bin )
endforeach(td)

######################
# Kernel Build Section
######################
set(KERN_SUFFIX "kernels.hsaco")

# Check if device-libs bitcode is following old or new layout
if(EXISTS "${BITCODE_DIR}/opencl.amdgcn.bc")
  set(BITCODE_ARGS "-nogpulib
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/opencl.amdgcn.bc
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ockl.amdgcn.bc
    -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ocml.amdgcn.bc")
else()
  set(BITCODE_ARGS "--hip-device-lib-path=${BITCODE_DIR}")
endif()

# Test Case Template example
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/test_case_template_kernels.cl")
build_sample_for_devices("test_case_template")

# P2P Memory Access
#set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
#set(CL_FILE_LIST "${KERNELS_DIR}/p2p_mem_access_kernels.cl")
#build_sample_for_devices("p2p_mem_access")

# Dispatch Time
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/dispatch_time_kernels.cl")
build_sample_for_devices("dispatch_time")

# gpuReadWrite
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/gpuReadWrite_kernels.cl")
build_sample_for_devices("gpuReadWrite")

# Vector Copy
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/vector_copy_kernel.cl")
build_sample_for_devices("vector_copy")

# Vector Add Debug Trap
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/vector_add_debug_trap_kernel.cl")
build_sample_for_devices("vector_add_debug_trap")

# Vector Add Memory Fault
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/vector_add_memory_fault_kernel.cl")
build_sample_for_devices("vector_add_memory_fault")

# atomic_add_kernels
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/atomicOperations_kernels.cl")
build_sample_for_devices("atomicOperations")

# Signal Operations
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/signal_operations.cl")
build_sample_for_devices("signal_operations")

# groupMemoryDynamic
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/groupMemoryDynamic_kernels.cl")
build_sample_for_devices("groupMemoryDynamic")

# groupMemoryDynamic
set(BITCODE_LIBS "${COMMON_BITCODE_LIBS}")
set(CL_FILE_LIST "${KERNELS_DIR}/cu_mask_kernels.cl")
build_sample_for_devices("cu_mask")

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Check for numa library and headers (same approach as libhsakmt)
find_package(PkgConfig)
find_package(NUMA)
if(NUMA_FOUND)
  set(NUMA_LIBS ${NUMA_LIBRARIES})
  message(STATUS "Found NUMA package: ${NUMA_LIBRARIES}")
  if(NUMA_INCLUDE_DIRS)
    include_directories(${NUMA_INCLUDE_DIRS})
    message(STATUS "NUMA include dirs: ${NUMA_INCLUDE_DIRS}")
  endif()
  set(HAVE_NUMA TRUE)
else()
  # Fallback to manual detection
  find_library(NUMA_LIBRARY NAMES numa)
  if(NUMA_LIBRARY)
    set(NUMA_LIBS ${NUMA_LIBRARY})
    message(STATUS "Found numa library (fallback): ${NUMA_LIBRARY}")
    set(HAVE_NUMA TRUE)
  else()
    message(WARNING "NUMA library not found. Building without numa support.")
    message(WARNING "  Install libnuma-dev (Debian/Ubuntu) or numactl-devel (RHEL/CentOS)")
    set(NUMA_LIBS "")
    set(HAVE_NUMA FALSE)
    # Exclude numa-dependent files from build
    list(FILTER performanceSources EXCLUDE REGEX ".*memory_async_copy.*\\.cc$")
  endif()
endif()

# Build rules
add_executable(${ROCRTST} ${performanceSources} ${functionalSources} ${negativeSources} ${stressSources}
                                           ${common_srcs} ${testCommonSources})

# Add HAVE_NUMA compile definition so code can conditionally compile NUMA features
target_compile_definitions(${ROCRTST} PRIVATE HAVE_NUMA=$<BOOL:${HAVE_NUMA}>)

# Find hwloc library (optional - only needed for certain performance tests)
# Try pkg-config first (for super-project builds)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(HWLOC QUIET hwloc)
  if(HWLOC_FOUND)
    set(HWLOC_LIBRARY ${HWLOC_LINK_LIBRARIES})
    message(STATUS "Found hwloc via pkg-config: ${HWLOC_LIBRARY}")
  endif()
endif()

# If not found via pkg-config, check for bundled version (any version)
if(NOT HWLOC_FOUND)
  # Use file(GLOB) to find any libhwloc.so* file regardless of version
  file(GLOB HWLOC_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/lib/libhwloc.so*")
  if(HWLOC_LIBRARY)
    list(GET HWLOC_LIBRARY 0 HWLOC_LIBRARY)  # Take first match if multiple versions exist
    set(HWLOC_FOUND TRUE)
    message(STATUS "Found bundled hwloc library: ${HWLOC_LIBRARY}")
  else()
    # Fall back to system hwloc
    find_library(HWLOC_LIBRARY NAMES hwloc)
    if(HWLOC_LIBRARY)
      set(HWLOC_FOUND TRUE)
      message(WARNING "Using system hwloc library: ${HWLOC_LIBRARY}")
      message(WARNING "  Consider using bundled version to avoid compatibility issues")
    endif()
  endif()
endif()

# Handle hwloc availability
if(HWLOC_FOUND)
  set(HWLOC_LIBS ${HWLOC_LIBRARY})
  set(HAVE_HWLOC TRUE)
  message(STATUS "hwloc support enabled")
else()
  message(FATAL_ERROR "hwloc library not found. Required for rocrtst build. Install libhwloc-dev (Debian/Ubuntu) or hwloc-devel (RHEL/CentOS)")
endif()

# Add HAVE_HWLOC compile definition
target_compile_definitions(${ROCRTST} PRIVATE HAVE_HWLOC=$<BOOL:${HAVE_HWLOC}>)

target_link_libraries(${ROCRTST} ${ROCRTST_LIBS} c stdc++ dl pthread rt ${NUMA_LIBS} ${HWLOC_LIBS})

#Build kernels
add_custom_target(rocrtst_kernels ALL DEPENDS ${HSACO_TARG_LIST})

#Build symlinks
add_custom_target(rocrtst_links ALL DEPENDS ${ROCRTST_LINKS_LIST} )

## Set RUNPATH to pickup local copy of hwloc
set_property(TARGET ${ROCRTST} PROPERTY INSTALL_RPATH "$ORIGIN;$ORIGIN/thirdparty/lib;$ORIGIN/../lib/rocrtst/thirdparty/lib" )
set_property(TARGET ${ROCRTST} PROPERTY LINK_FLAGS "-Wl,--enable-new-dtags")

install(TARGETS ${ROCRTST}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/lib DESTINATION lib/rocrtst )

include ( CPack )
