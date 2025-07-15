################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2021, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

cmake_minimum_required ( VERSION 3.7 )

## Clear target dependency data.
## Needed to allow UI transitions between static and dynamic builds.
## Need an update to CMake 3.12 to remove this hack. See CMake policy change CMP0073.
unset ( hsa-runtime64_LIB_DEPENDS CACHE )

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

## Set core runtime module name and project name.
set ( CORE_RUNTIME_NAME "hsa-runtime64" )
set ( CORE_RUNTIME_TARGET "${CORE_RUNTIME_NAME}" )
set ( CORE_RUNTIME_LIBRARY "lib${CORE_RUNTIME_TARGET}" )

## Set project name
project( ${CORE_RUNTIME_TARGET} )

## Utilty functions
list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" )
include ( utils )
include ( hsa_common )
include ( GNUInstallDirs )

## Adjust target name for static builds
## Original name will be an interface target that adds --whole-archive linker options around the target.
if( NOT ${BUILD_SHARED_LIBS} )
  set ( CORE_RUNTIME_TARGET "${CORE_RUNTIME_TARGET}_static" )
endif()

# Optionally, build HSA Runtime with ccache.
set(ROCM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if (ROCM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if (CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find ccache. Falling back to real compiler")
  endif() # if (CCACHE_PROGRAM)
endif() # if (ROCM_CCACHE_BUILD)

## Find external dependencies.
find_package(PkgConfig)
find_package(LibElf REQUIRED)

pkg_check_modules(drm REQUIRED IMPORTED_TARGET libdrm)

## Create the rocr target.
add_library( ${CORE_RUNTIME_TARGET} "" )

## Enforce uniform output file naming.
set_property(TARGET  ${CORE_RUNTIME_TARGET} PROPERTY OUTPUT_NAME ${CORE_RUNTIME_NAME} )

## Compiler preproc definitions.
target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE "${HSA_COMMON_DEFS}" __linux__ HSA_EXPORT=1 HSA_EXPORT_FINALIZER=1 HSA_EXPORT_IMAGES=1 HSA_DEPRECATED=
ROCR_BUILD_ID="${PACKAGE_VERSION_STRING}-${VERSION_JOB}-${VERSION_HASH}" )

## Check for memfd_create syscall
include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS ( "__NR_memfd_create" "sys/syscall.h" HAVE_MEMFD_CREATE )
if ( HAVE_MEMFD_CREATE )
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HAVE_MEMFD_CREATE )
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

## Set include directories for ROCr runtime
target_include_directories( ${CORE_RUNTIME_TARGET}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/libamdhsacode
  ${CMAKE_CURRENT_BINARY_DIR}/core/runtime/trap_handler
  ${CMAKE_CURRENT_BINARY_DIR}/core/runtime/blit_shaders)


## ------------------------- Linux Compiler and Linker options -------------------------
set ( HSA_CXX_FLAGS ${HSA_COMMON_CXX_FLAGS} -fexceptions -fno-rtti -fvisibility=hidden -Wno-error=missing-braces -Wno-error=sign-compare -Wno-sign-compare -Wno-write-strings -Wno-conversion-null -fno-math-errno -fno-threadsafe-statics -fmerge-all-constants -fms-extensions -Wno-error=comment -Wno-comment -Wno-error=pointer-arith -Wno-pointer-arith -Wno-error=unused-variable -Wno-error=unused-function )

## Extra x86 specific settings
if ( CMAKE_SYSTEM_PROCESSOR MATCHES "i?86|x86_64|amd64|AMD64" )
  set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -mmwaitx )
endif()

## Extra image settings - audit!
set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-deprecated-declarations )

if ( CMAKE_COMPILER_IS_GNUCXX )
    set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=unused-but-set-variable)
endif ()
if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=self-assign)
  if( ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13)
    set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=unused-but-set-variable)
  endif()
endif()

set ( DRVDEF "${CMAKE_CURRENT_SOURCE_DIR}/hsacore.so.def" )
set ( LNKSCR "hsacore.so.link" )
set ( HSA_SHARED_LINK_FLAGS "-Wl,-Bdynamic -Wl,-z,noexecstack -Wl,${CMAKE_CURRENT_SOURCE_DIR}/${LNKSCR} -Wl,--version-script=${DRVDEF} -Wl,--enable-new-dtags" )

target_compile_options(${CORE_RUNTIME_TARGET} PRIVATE ${HSA_CXX_FLAGS})
#target_link_options not available prior to CMake 3.13
set_property(TARGET ${CORE_RUNTIME_TARGET} PROPERTY LINK_FLAGS ${HSA_SHARED_LINK_FLAGS})
##  -------------------------  End Compiler and Linker options ----------------------------

## Source files.
set ( SRCS core/driver/driver.cpp
           core/driver/kfd/amd_kfd_driver.cpp
           core/driver/xdna/amd_xdna_driver.cpp
           core/util/lnx/os_linux.cpp
           core/util/small_heap.cpp
           core/util/timer.cpp
           core/util/flag.cpp
           core/runtime/amd_aie_agent.cpp
           core/runtime/amd_aie_aql_queue.cpp
           core/runtime/amd_blit_kernel.cpp
           core/runtime/amd_blit_sdma.cpp
           core/runtime/amd_cpu_agent.cpp
           core/runtime/amd_gpu_agent.cpp
           core/runtime/amd_hsa_loader.cpp
           core/runtime/amd_aql_queue.cpp
           core/runtime/amd_loader_context.cpp
           core/runtime/hsa_ven_amd_loader.cpp
           core/runtime/amd_memory_region.cpp
           core/runtime/amd_filter_device.cpp
           core/runtime/amd_topology.cpp
           core/runtime/default_signal.cpp
           core/runtime/host_queue.cpp
           core/runtime/hsa.cpp
           core/runtime/hsa_api_trace.cpp
           core/runtime/hsa_ext_amd.cpp
           core/runtime/hsa_ext_interface.cpp
           core/runtime/interrupt_signal.cpp
           core/runtime/intercept_queue.cpp
           core/runtime/ipc_signal.cpp
           core/runtime/isa.cpp
           core/runtime/runtime.cpp
           core/runtime/signal.cpp
           core/runtime/queue.cpp
           core/runtime/cache.cpp
           core/runtime/svm_profiler.cpp
           core/runtime/thunk_loader.cpp
           core/common/hsa_table_interface.cpp
           loader/executable.cpp
           libamdhsacode/amd_elf_image.cpp
           libamdhsacode/amd_hsa_code_util.cpp
           libamdhsacode/amd_hsa_locks.cpp
           libamdhsacode/amd_options.cpp
           libamdhsacode/amd_hsa_code.cpp
           libamdhsacode/amd_core_dump.cpp )

if ( BUILD_THUNK_VIRTIO )
  list(APPEND SRCS core/driver/virtio/amd_kfd_virtio_driver.cpp)
  target_compile_definitions(hsa-runtime64 PRIVATE HSAKMT_VIRTIO_ENABLED=1)
endif()

target_sources( ${CORE_RUNTIME_TARGET} PRIVATE ${SRCS} )

## Depend on trap handler target.
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime/trap_handler )
add_dependencies( ${CORE_RUNTIME_TARGET} amd_trap_handler_v2 )

## Depend on blit shader target.
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime/blit_shaders )
add_dependencies( ${CORE_RUNTIME_TARGET} amd_blit_shaders_v2)

option(PC_SAMPLING_SUPPORT "Enable PC Sampling Support" ON)

if (${PC_SAMPLING_SUPPORT})
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HSA_PC_SAMPLING_SUPPORT)

  set( PCS_SRCS pcs/hsa_ven_amd_pc_sampling.cpp pcs/pcs_runtime.cpp )

  target_sources( ${CORE_RUNTIME_TARGET} PRIVATE ${PCS_SRCS} )
endif()

if ( NOT DEFINED IMAGE_SUPPORT AND CMAKE_SYSTEM_PROCESSOR MATCHES "i?86|x86_64|amd64|AMD64|loongarch64" )
  set ( IMAGE_SUPPORT ON )
endif()
set ( IMAGE_SUPPORT ${IMAGE_SUPPORT} CACHE BOOL "Build with image support (default: ON for x86, OFF elsewise)." )

## Optional image module defintions.
if(${IMAGE_SUPPORT})
  ## Image definitons - audit!
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE
    HSA_IMAGE_SUPPORT
    UNIX_OS
    LINUX
    AMD_INTERNAL_BUILD
    BRAHMA_BUILD=1 )

  set ( IMAGE_SRCS image/addrlib/src/addrinterface.cpp
                   image/addrlib/src/core/coord.cpp
                   image/addrlib/src/core/addrlib.cpp
                   image/addrlib/src/core/addrlib1.cpp
                   image/addrlib/src/core/addrlib2.cpp
                   image/addrlib/src/core/addrlib3.cpp
                   image/addrlib/src/core/addrobject.cpp
                   image/addrlib/src/core/addrelemlib.cpp
                   image/addrlib/src/gfx9/gfx9addrlib.cpp
                   image/addrlib/src/gfx10/gfx10addrlib.cpp
                   image/addrlib/src/gfx11/gfx11addrlib.cpp
                   image/addrlib/src/gfx12/gfx12addrlib.cpp
                   image/device_info.cpp
                   image/hsa_ext_image.cpp
                   image/image_runtime.cpp
                   image/image_manager.cpp
                   image/image_manager_kv.cpp
                   image/image_manager_ai.cpp
                   image/image_manager_nv.cpp
                   image/image_manager_gfx11.cpp
                   image/image_manager_gfx12.cpp
                   image/image_lut_kv.cpp
                   image/image_lut_gfx11.cpp
                   image/blit_object_gfx7xx.cpp
                   image/blit_object_gfx8xx.cpp
                   image/blit_object_gfx9xx.cpp
                   image/blit_kernel.cpp
                   ${CMAKE_CURRENT_BINARY_DIR}/image/blit_src/opencl_blit_objects.cpp )

  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/image/blit_src/opencl_blit_objects.cpp PROPERTIES GENERATED TRUE)

  target_include_directories( ${CORE_RUNTIME_TARGET}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/image
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/r800
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/gfx9
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/gfx10
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/gfx11
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/gfx12
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/chip/r800
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/chip/gfx9
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/chip/gfx10
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/chip/gfx11
    ${CMAKE_CURRENT_SOURCE_DIR}/image/addrlib/src/chip/gfx12 )

  target_sources( ${CORE_RUNTIME_TARGET} PRIVATE ${IMAGE_SRCS} )

  ## Depend on blit kernel target.
  add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/image/blit_src )
  add_dependencies( ${CORE_RUNTIME_TARGET} opencl_blit_objects )

endif()

target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE elf::elf dl pthread rt )
# For static package rocprofiler-register dependency is not required
# Link to hsakmt target for shared library builds
# Link to hsakmt-staticdrm target for static library builds
if( BUILD_SHARED_LIBS )
  target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt::hsakmt PkgConfig::drm)
  if( BUILD_THUNK_VIRTIO )
    message(STATUS "Building with virtio support")
    target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt_virtio)
  endif()
  find_package(rocprofiler-register)
  if(rocprofiler-register_FOUND)
    target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HSA_ROCPROFILER_REGISTER=1
                                                              HSA_VERSION_MAJOR=${VERSION_MAJOR}
                                                              HSA_VERSION_MINOR=${VERSION_MINOR}
                                                              HSA_VERSION_PATCH=${VERSION_PATCH})
    target_link_libraries(${CORE_RUNTIME_TARGET} PRIVATE rocprofiler-register::rocprofiler-register)
    set(HSA_DEP_ROCPROFILER_REGISTER ON CACHE INTERNAL "")
  else()
    set(HSA_DEP_ROCPROFILER_REGISTER OFF CACHE INTERNAL "")
  endif() # end rocprofiler-register_FOUND
else()
  include_directories(${drm_INCLUDE_DIRS})
  target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt-staticdrm::hsakmt-staticdrm)
endif()#end BUILD_SHARED_LIBS

## Set the VERSION and SOVERSION values
set_property ( TARGET ${CORE_RUNTIME_TARGET} PROPERTY VERSION "${SO_VERSION_STRING}" )
set_property ( TARGET ${CORE_RUNTIME_TARGET} PROPERTY SOVERSION "${VERSION_MAJOR}" )

## Add the public interface export target if doing a static build.
## Bind ROCr dependencies to the interface target rather than to the source build
## target so that -Wl,--whole-archive is tightly applied.  Requires binding
## indirectly to the source build taret.
if( NOT ${BUILD_SHARED_LIBS} )
  add_library(${CORE_RUNTIME_NAME} INTERFACE)

  ## Bind to source build target interface but not its link requirements.
  target_include_directories( ${CORE_RUNTIME_NAME} INTERFACE $<TARGET_PROPERTY:${CORE_RUNTIME_NAME}::${CORE_RUNTIME_TARGET},INTERFACE_INCLUDE_DIRECTORIES> )
  target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE -Wl,$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}/${LNKSCR}
    -Wl,--whole-archive $<TARGET_FILE:${CORE_RUNTIME_NAME}::${CORE_RUNTIME_TARGET}> -Wl,--no-whole-archive)
  add_dependencies( ${CORE_RUNTIME_NAME} ${CORE_RUNTIME_TARGET} )

  ## Add external link requirements.
  target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE hsakmt-staticdrm::hsakmt-staticdrm )
  target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE elf::elf dl pthread rt )

  install ( TARGETS ${CORE_RUNTIME_NAME} EXPORT ${CORE_RUNTIME_NAME}Targets )
endif()

## Set install information
# Installs binaries and exports the library usage data to ${HSAKMT_TARGET}Targets
install ( TARGETS ${CORE_RUNTIME_TARGET} EXPORT ${CORE_RUNTIME_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary )
install ( TARGETS ${CORE_RUNTIME_TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )

# Install license
if(ENABLE_ASAN_PACKAGING)
   install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan )
endif()
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary )

# Install public headers
install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hsa COMPONENT dev )

## Configure and install package config file
# Record our usage data for clients find_package calls.
install ( EXPORT ${CORE_RUNTIME_NAME}Targets
  FILE ${CORE_RUNTIME_NAME}Targets.cmake
  NAMESPACE ${CORE_RUNTIME_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
  COMPONENT dev)

# Adds the target alias hsa-runtime64::hsa-runtime64 to the local cmake cache.
# This isn't necessary today.  It's harmless preparation for some
# hypothetical future in which the we might be included by add_subdirectory()
# in some other project's cmake file.  It allows uniform use of find_package
# and target_link_library() without regard to whether a target is external or
# a subdirectory of the current build.
add_library( ${CORE_RUNTIME_NAME}::${CORE_RUNTIME_NAME} ALIAS ${CORE_RUNTIME_NAME} )

# Create cmake configuration files
include(CMakePackageConfigHelpers)

configure_package_config_file(${CORE_RUNTIME_NAME}-config.cmake.in
  ${CORE_RUNTIME_NAME}-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME} )

write_basic_package_version_file(${CORE_RUNTIME_NAME}-config-version.cmake
  VERSION ${SO_VERSION_STRING} COMPATIBILITY AnyNewerVersion )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CORE_RUNTIME_NAME}-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${CORE_RUNTIME_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
  COMPONENT dev)

# Install build files needed only when using a static build.
if( NOT ${BUILD_SHARED_LIBS} )
  # libelf find package module
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindLibElf.cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/COPYING-CMAKE-SCRIPTS
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
    COMPONENT dev)
  # Linker script (defines function aliases)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${LNKSCR}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
    COMPONENT dev)
endif()
