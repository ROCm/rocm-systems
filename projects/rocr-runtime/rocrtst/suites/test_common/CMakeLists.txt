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

if(WIN32)
#############################
# WINDOWS PLATFORM SECTION
#############################
  # Windows-specific compiler flags and definitions
  add_definitions(-D_WIN32_WINNT=0x0600)  # Windows Vista or later
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)

  # Check for explicit HSA library path
  if(DEFINED HSA_LIBRARY AND EXISTS "${HSA_LIBRARY}")
    message(STATUS "Using explicit HSA library: ${HSA_LIBRARY}")
  else()
    message(FATAL_ERROR "HSA_LIBRARY path must be provided for Windows builds")
  endif()

  # C++20 standard for designated initializers in MSVC
  set(CMAKE_CXX_STANDARD 20)

  if(BUILD_SHARED_LIBS)
    # Dynamic runtime for shared libraries
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")     # Dynamic runtime for debug
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")  # Dynamic runtime for release
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")         # Dynamic runtime for debug
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD")      # Dynamic runtime for release
  else()
    # Static runtime for static libraries
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")     # Static runtime for debug
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")  # Static runtime for release
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")         # Static runtime for debug
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")      # Static runtime for release
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")  # Exception handling
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")     # Warning level 3
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj") # Support large object files

  # Disable specific warnings for Windows
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996") # Disable deprecated function warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267") # Disable size_t conversion warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244") # Disable conversion warnings
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4200") # Disable zero-sized array warnings

  # Handle LLVM/OpenCL paths for Windows
  if (DEFINED LLVM_DIR)
    set(CLANG ${LLVM_DIR}/bin/clang.exe)
    if (NOT EXISTS ${CLANG})
      message("ERROR: path to clang (${CLANG}) is not valid. Is define LLVM_DIR correct?")
      return()
    endif()
  else()
    message("ERROR: LLVM_DIR define is not set. Kernels will not be built on Windows.")
    return()
  endif()

  if (DEFINED OPENCL_DIR)
    set(OPENCL_INC_DIR ${OPENCL_DIR}/include)
    set(OPENCL_LIB_DIR ${OPENCL_DIR}/lib)
  endif()

  if (DEFINED OPENCL_VER)
    set(OPENCL_VER ${OPENCL_VER})
  else()
    message("OPENCL_VER define is not set. Using default")
    set(OPENCL_VER "2.0")
  endif()

  # Check for opencl-c.h header file
  if(NOT EXISTS "${OPENCL_INC_DIR}/opencl-c.h")
    # Automatically find all clang version directories first
    set(POSSIBLE_OPENCL_PATHS "")

    file(GLOB CLANG_DIRS "${OPENCL_DIR}/lib/clang/*/include")
    foreach(dir ${CLANG_DIRS})
      if(EXISTS "${dir}/opencl-c.h")
        list(APPEND POSSIBLE_OPENCL_PATHS "${dir}")
      endif()
    endforeach()

    # Fallback paths in case glob doesn't find the PATH
    list(APPEND POSSIBLE_OPENCL_PATHS
      "${OPENCL_DIR}/lib/clang/25/include"
      "${OPENCL_DIR}/lib/clang/24/include"
      "${OPENCL_DIR}/lib/clang/23/include"
      "${OPENCL_DIR}/lib/clang/22/include"
      "${OPENCL_DIR}/lib/clang/21/include"
      "${OPENCL_DIR}/lib/clang/20/include"
      "${OPENCL_DIR}/lib/clang/19/include"
      "${OPENCL_DIR}/lib/clang/18/include"
      "${OPENCL_DIR}/lib/clang/17/include"
    )

    if(DEFINED ENV{LLVM_PROJECT_ROOT})
      list(APPEND POSSIBLE_OPENCL_PATHS "$ENV{LLVM_PROJECT_ROOT}/clang/lib/Headers/")
    endif()

    foreach(path ${POSSIBLE_OPENCL_PATHS})
      if(EXISTS "${path}/opencl-c.h")
        set(OPENCL_INC_DIR "${path}")
        message(STATUS "Found opencl-c.h at: ${OPENCL_INC_DIR}")
        break()
      endif()
    endforeach()

    if(NOT EXISTS "${OPENCL_INC_DIR}/opencl-c.h")
      message(WARNING "opencl-c.h not found in any expected location.")
    endif()
  endif()

  if (NOT DEFINED TARGET_DEVICES)
    message("No targets devices provided on command line")
    message("  e.g., cmake -DTARGET_DEVICES=\"gfx803;gfx900;gfx...\" ..")
    message("  Using default target of ${DEFAULT_TARGETS}")
    list(APPEND TARGET_DEVICES ${DEFAULT_TARGETS})
  endif()

  string(TOLOWER "${ROCRTST_BLD_TYPE}" tmp)
  if("${tmp}" STREQUAL release)
    set(BUILD_TYPE "Release")
    set(ISDEBUG 0)
  else()
    set(BUILD_TYPE "Debug")
    set(ISDEBUG 1)
  endif()

  if(${BUILD_TYPE} STREQUAL "Debug")
    add_definitions(-DDEBUG)
  endif()

  if(${EMULATOR_BUILD})
    add_definitions(-DROCRTST_EMULATOR_BUILD=1)
  endif()

  add_definitions(-DLITTLEENDIAN_CPU=1)
  MESSAGE("ISDEBUG STEP:Done")

  set(ROCRTST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

  # Bitcode directory setup - use same logic as Linux
  find_path(BITCODE_DIR NAMES "opencl.bc" "opencl.amdgcn.bc"
    PATHS
      "${ROCM_DIR}/amdgcn/bitcode"
      "${ROCM_DIR}/lib/bitcode"
      "${ROCM_DIR}/lib"
      "${ROCM_DIR}/lib/x86_64/bitcode"
      "${OPENCL_DIR}/amdgcn/bitcode"
      "${OPENCL_DIR}/lib/x86_64/bitcode"
      "${LLVM_DIR}/../lib/bitcode"
      "${CMAKE_PREFIX_PATH}/amdgcn/bitcode"
      "${CMAKE_PREFIX_PATH}/lib/bitcode"
      "${CMAKE_PREFIX_PATH}/lib/x86_64/bitcode"
  )

  # Check if device-libs bitcode is following old or new layout
  if(EXISTS "${BITCODE_DIR}/opencl.amdgcn.bc")
    set(BITCODE_ARGS "-nogpulib -fshort-wchar
      -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/opencl.amdgcn.bc
      -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ockl.amdgcn.bc
      -Xclang -mlink-bitcode-file -Xclang ${BITCODE_DIR}/ocml.amdgcn.bc")
  else()
    set(BITCODE_ARGS "--hip-device-lib-path=${BITCODE_DIR} -fshort-wchar")
  endif()

  # Windows Build Configuration
  message("Build Configuration:")
  message("-------------IS64BIT: " ${IS64BIT})
  message("-----------BuildType: " ${BUILD_TYPE})
  message("------------Compiler: " ${CMAKE_CXX_COMPILER})
  message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
  message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
  message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
  message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
  message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
  message("------Target Devices: " ${TARGET_DEVICES})
  if(EXISTS "${CLANG}")
    message("----------Clang path: " ${CLANG})
  endif()
  message("----------OpenCL Dir: " ${OPENCL_DIR})
  message("-------OpenCL version " ${OPENCL_VER})

  set(KERNELS_DIR ${PROJECT_SOURCE_DIR}/kernels)
  set(COMMON_BITCODE_LIBS "")
  set(HSACO_TARG_LIST "")

  # Set Name for Google Test Framework and build it as a static library
  set(GOOGLE_TEST_FRWK_NAME "google-test-frwk64")
  add_subdirectory(${ROCRTST_ROOT}/gtest "${PROJECT_BINARY_DIR}/gtest" EXCLUDE_FROM_ALL)
  set (ROCRTST_LIBS ${ROCRTST_LIBS} ${GOOGLE_TEST_FRWK_NAME})

  message("ROCRTST_LIBS SET STEP:Done")

  # Other source directories
  aux_source_directory(${ROCRTST_ROOT}/common common_srcs)

  # Add Windows compatibility sources
  list(APPEND common_srcs ${ROCRTST_ROOT}/common/windows/windows_compat.cc)
  list(APPEND common_srcs ${ROCRTST_ROOT}/common/windows/mman.cc)

  # Extend the list of libraries to be used for linking ROC Perf Apps
  set(ROCRTST_LIBS ${ROCRTST_LIBS} ${HSA_LIBRARY})

  # Add additional ROCm libraries and Windows system libraries
  set(ROCRTST_LIBS ${ROCRTST_LIBS}
    ${AMD_ROCR_EXTRA_LIB}
    ws2_32 advapi32 psapi)

  # Set Name for rocrtst
  message(${ROCRTST_LIBS})
  set(ROCRTST "rocrtst64")

  # Source files for building rocrtst
  aux_source_directory(${ROCRTST_ROOT}/suites/performance performanceSources)
  aux_source_directory(${ROCRTST_ROOT}/suites/functional functionalSources)
  aux_source_directory(${ROCRTST_ROOT}/suites/negative negativeSources)
  aux_source_directory(${ROCRTST_ROOT}/suites/stress stressSources)
  aux_source_directory(${ROCRTST_ROOT}/suites/test_common testCommonSources)

  include_directories(${ROCRTST_ROOT}/suites/test_common)
  include_directories(${ROCRTST_ROOT}/common)
  include_directories(${ROCRTST_ROOT}/gtest/include)
  include_directories(${ROCRTST_ROOT})
  include_directories(${HSA_PATH}/include)
  include_directories(${ROCRTST_ROOT}/common/amd_smi)
  include_directories(${ROCRTST_ROOT}/common/windows)
  include_directories(${ROCRTST_ROOT}/thirdparty/include)

  # Kernel(shader) build function specifically for Windows
  function(build_kernel S_NAME TARG_DEV)
    set(KERNEL_DIR "${PROJECT_BINARY_DIR}/${TARG_DEV}")
    file(MAKE_DIRECTORY ${KERNEL_DIR})
    set(SNAME_KERNEL "${S_NAME}_kernels.hsaco")

    set(TARG_NAME "${S_NAME}_hsaco.${TARG_DEV}")
    set(HSACO_TARG_LIST ${HSACO_TARG_LIST} "${KERNEL_DIR}/${SNAME_KERNEL}" PARENT_SCOPE)
    string(SUBSTRING ${TARG_DEV} 3 -1 gfxNum)

    separate_arguments(CLANG_ARG_LIST WINDOWS_COMMAND
     "-D ROCRTST_GPU=0x${gfxNum} -x cl -target amdgcn-amd-amdhsa -include \"${OPENCL_INC_DIR}/opencl-c.h\" -mcpu=${TARG_DEV} ${BITCODE_ARGS} -cl-std=CL${OPENCL_VER} -mcode-object-version=4 ${CL_FILE_LIST} -o \"${KERNEL_DIR}/${SNAME_KERNEL}\"")
    add_custom_command(OUTPUT "${KERNEL_DIR}/${SNAME_KERNEL}" COMMAND ${CLANG} ${CLANG_ARG_LIST} DEPENDS ${CL_FILE_LIST} ${CLANG} COMMENT "BUILDING ${KERNEL_DIR}/${SNAME_KERNEL}" VERBATIM)
  endfunction(build_kernel)

  function(build_sample_for_devices S_NAME)
    foreach(TARG_DEV ${TARGET_DEVICES})
      build_kernel(${S_NAME} ${TARG_DEV})
    endforeach(TARG_DEV)
    set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)
  endfunction(build_sample_for_devices)

  # Make directories for each possible target device (no symlinks on Windows)
  foreach(td ${TARGET_DEVICES})
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/${td})
    install ( DIRECTORY ${PROJECT_BINARY_DIR}/${td} DESTINATION bin )
  endforeach(td)

  # Build kernels for all test cases
  set(CL_FILE_LIST "${KERNELS_DIR}/test_case_template_kernels.cl")
  build_sample_for_devices("test_case_template")

  set(CL_FILE_LIST "${KERNELS_DIR}/dispatch_time_kernels.cl")
  build_sample_for_devices("dispatch_time")

  set(CL_FILE_LIST "${KERNELS_DIR}/gpuReadWrite_kernels.cl")
  build_sample_for_devices("gpuReadWrite")

  set(CL_FILE_LIST "${KERNELS_DIR}/vector_add_debug_trap_kernel.cl")
  build_sample_for_devices("vector_add_debug_trap")

  set(CL_FILE_LIST "${KERNELS_DIR}/vector_add_memory_fault_kernel.cl")
  build_sample_for_devices("vector_add_memory_fault")

  set(CL_FILE_LIST "${KERNELS_DIR}/atomicOperations_kernels.cl")
  build_sample_for_devices("atomicOperations")

  set(CL_FILE_LIST "${KERNELS_DIR}/signal_operations.cl")
  build_sample_for_devices("signal_operations")

  set(CL_FILE_LIST "${KERNELS_DIR}/groupMemoryDynamic_kernels.cl")
  build_sample_for_devices("groupMemoryDynamic")

  set(CL_FILE_LIST "${KERNELS_DIR}/cu_mask_kernels.cl")
  build_sample_for_devices("cu_mask")

  set(CL_FILE_LIST "${KERNELS_DIR}/vector_copy_kernel.cl")
  build_sample_for_devices("vector_copy")

  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

  # Build rules
  add_executable(${ROCRTST} ${performanceSources} ${functionalSources} ${negativeSources} ${stressSources}
                                             ${common_srcs} ${testCommonSources})

  target_link_libraries(${ROCRTST} ${ROCRTST_LIBS})

  #Build kernels
  add_custom_target(rocrtst_kernels ALL DEPENDS ${HSACO_TARG_LIST})

  install(TARGETS ${ROCRTST}
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)

  install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/lib DESTINATION lib/rocrtst )
else()
#############################
# LINUX PLATFORM SECTION
#############################
# Process input variables

# Required Defines first:
find_package(hsa-runtime64 REQUIRED)
find_package(amd_smi REQUIRED)

set (ONLY64STR "64")

if (DEFINED LLVM_DIR)
  set(CLANG ${LLVM_DIR}/clang)
  if (NOT EXISTS ${CLANG})
    message("ERROR: path to clang (${CLANG}) is not valid. Is define LLVM_DIR correct?")
    return()
  endif()
else()
  message("WARNING: LLVM_DIR define is not set. Kernels will not be built.")
endif()

if (DEFINED OPENCL_DIR)
  set(OPENCL_INC_DIR ${OPENCL_DIR}/include)
  set(OPENCL_LIB_DIR ${OPENCL_DIR}/lib)
else()
  message("WARNING: OPENCL_DIR define is not set. Kernels will not be built.")
endif()

if (DEFINED OPENCL_VER)
  set(OPENCL_VER ${OPENCL_VER})
else()
  message("OPENCL_VER define is not set. Using default")
  set(OPENCL_VER "2.0")
endif()

if(NOT EXISTS "${OPENCL_INC_DIR}/opencl-c.h")
  if(DEFINED ENV{LLVM_PROJECT_ROOT})
    set(OPENCL_INC_DIR "$ENV{LLVM_PROJECT_ROOT}/clang/lib/Headers/")
  else()
    set(OPENCL_INC_DIR "${OPENCL_DIR}/../../../external/llvm-project/clang/lib/Headers/")
  endif()
  if(NOT EXISTS "${OPENCL_INC_DIR}/opencl-c.h")
    message(WARNING "opencl-c.h not found.")
  endif()
endif()

if (NOT DEFINED TARGET_DEVICES)
  message("No targets devices provided on command line")
  message("  e.g., cmake -DTARGET_DEVICES=\"gfx803;gfx900;gfx...\" ..")
  message("  Using default target of ${DEFAULT_TARGETS}")
  list(APPEND TARGET_DEVICES ${DEFAULT_TARGETS})
endif()

string(TOLOWER "${ROCRTST_BLD_TYPE}" tmp)
if("${tmp}" STREQUAL release)
  set(BUILD_TYPE "Release")
  set(ISDEBUG 0)
else()
  set(BUILD_TYPE "Debug")
  set(ISDEBUG 1)
endif()

find_path(BITCODE_DIR NAMES "opencl.bc" "opencl.amdgcn.bc"
  PATHS
    "${ROCM_DIR}/amdgcn/bitcode"
    "${ROCM_DIR}/lib/bitcode"
    "${ROCM_DIR}/lib"
    "${ROCM_DIR}/lib/x86_64/bitcode"
    "${OPENCL_DIR}/amdgcn/bitcode"
    "${OPENCL_DIR}/lib/x86_64/bitcode"
    "${LLVM_DIR}/../lib/bitcode"
    "${CMAKE_PREFIX_PATH}/amdgcn/bitcode"
    "${CMAKE_PREFIX_PATH}/lib/bitcode"
    "${CMAKE_PREFIX_PATH}/lib/x86_64/bitcode")

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
message("----------OpenCL Dir: " ${OPENCL_DIR})
message("-------OpenCL version " ${OPENCL_VER})
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
# Compiler options
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

# Build rules
add_executable(${ROCRTST} ${performanceSources} ${functionalSources} ${negativeSources} ${stressSources}
                                           ${common_srcs} ${testCommonSources})

target_link_libraries(${ROCRTST} ${ROCRTST_LIBS} c stdc++ dl pthread rt numa ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/lib/libhwloc.so.5)

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
endif()  # End of WIN32/Linux platform selection
