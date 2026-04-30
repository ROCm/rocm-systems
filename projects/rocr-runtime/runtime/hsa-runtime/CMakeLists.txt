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
if (UNIX)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 20)
endif()

## Set core runtime module name and project name.
set ( CORE_RUNTIME_NAME "hsa-runtime64" )
set ( CORE_RUNTIME_TARGET "${CORE_RUNTIME_NAME}" )
set ( CORE_RUNTIME_LIBRARY "lib${CORE_RUNTIME_TARGET}" )

## Set project name
project( hsa-rocr )

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

find_package(LibElf CONFIG)
if(NOT LibElf_FOUND)
  find_package(LibElf REQUIRED)
endif()

## Find AMD CUID library.
set(AMDCUID_ROOT "/opt/amdcuid" CACHE PATH "Path to AMDCUID installation")
find_library(AMDCUID_LIB
  NAMES amdcuid_shared
  PATHS ${AMDCUID_ROOT}/lib
  NO_DEFAULT_PATH
)
find_path(AMDCUID_INCLUDE_DIR
  NAMES amd_cuid.h
  PATHS ${AMDCUID_ROOT}/include
  NO_DEFAULT_PATH
)
if (AMDCUID_LIB AND AMDCUID_INCLUDE_DIR) 
  message(STATUS "Found AMDCUID lib: ${AMDCUID_LIB}")
  message(STATUS "Found AMDCUID header: ${AMDCUID_INCLUDE_DIR}")
  set(AMDCUID_FOUND TRUE)
else()
  message(WARNING "AMDCUID Library not found")
  set(AMDCUID_FOUND FALSE)
endif()

## Create the rocr target.
add_library( ${CORE_RUNTIME_TARGET} "" )

## Enable parallel compilation on MSVC.
if(MSVC)
  if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    target_compile_options(${CORE_RUNTIME_TARGET} PRIVATE /MP$ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  endif()
endif()

if (UNIX)
  pkg_check_modules(drm REQUIRED IMPORTED_TARGET libdrm)
else()
  target_include_directories(${CORE_RUNTIME_TARGET} PRIVATE ${LIBELF_INCLUDE_DIR})
  if (${BUILD_SHARED_LIBS})
    target_link_libraries(${CORE_RUNTIME_TARGET} PRIVATE oclelf)
  endif()
endif()
## Enforce uniform output file naming.
set_property(TARGET  ${CORE_RUNTIME_TARGET} PROPERTY OUTPUT_NAME ${CORE_RUNTIME_NAME} )

## Compiler preproc definitions.
if (UNIX)
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
else()
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE "${HSA_COMMON_DEFS}" AMD_LIBELF=1 HSA_EXPORT=1 HSA_EXPORT_FINALIZER=1 HSA_EXPORT_IMAGES=1 HSA_DEPRECATED=
    ROCR_BUILD_ID="${PACKAGE_VERSION_STRING}-${VERSION_JOB}-${VERSION_HASH}")
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
if (UNIX)
  set ( HSA_CXX_FLAGS ${HSA_COMMON_CXX_FLAGS} -fexceptions -fno-rtti -fvisibility=hidden -Wno-error=missing-braces -Wno-error=sign-compare -Wno-sign-compare -Wno-write-strings -Wno-conversion-null -fno-math-errno -fno-threadsafe-statics -fmerge-all-constants -fms-extensions -Wno-error=comment -Wno-comment -Wno-error=pointer-arith -Wno-pointer-arith -Wno-error=unused-variable -Wno-error=unused-function )

  ## Extra x86 specific settings
  if ( CMAKE_SYSTEM_PROCESSOR MATCHES "i?86|x86_64|amd64|AMD64" )
    set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -mmwaitx )
  endif()

  ## Extra image settings - audit!
  set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-deprecated-declarations )
endif()

if ( CMAKE_COMPILER_IS_GNUCXX )
    set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=unused-but-set-variable)
endif ()
if ( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=self-assign)
  if( ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13)
    set ( HSA_CXX_FLAGS ${HSA_CXX_FLAGS} -Wno-error=unused-but-set-variable)
  endif()
endif()

if (UNIX)
  set ( LNKSCR "hsacore.so.link" )
  set ( DRVDEF "${CMAKE_CURRENT_SOURCE_DIR}/hsacore.so.def" )
  set(HSA_SHARED_LINK_FLAGS "-Wl,-Bdynamic -Wl,-z,noexecstack -Wl,${CMAKE_CURRENT_SOURCE_DIR}/${LNKSCR} -Wl,--version-script=${DRVDEF} -Wl,--enable-new-dtags")
else()
  target_sources(${CORE_RUNTIME_TARGET} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hsacore.dll.def")
  set(HSA_SHARED_LINK_FLAGS "")
endif()

target_compile_options(${CORE_RUNTIME_TARGET} PRIVATE ${HSA_CXX_FLAGS})
#target_link_options not available prior to CMake 3.13
set_property(TARGET ${CORE_RUNTIME_TARGET} PROPERTY LINK_FLAGS ${HSA_SHARED_LINK_FLAGS})
##  -------------------------  End Compiler and Linker options ----------------------------

## Source files.
set ( SRCS core/driver/driver.cpp
           core/driver/kfd/amd_kfd_driver.cpp
           core/util/small_heap.cpp
           core/util/timer.cpp
           core/util/flag.cpp
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
           core/runtime/counted_queue_manager.cpp
           core/common/hsa_table_interface.cpp
           loader/executable.cpp
           libamdhsacode/amd_elf_image.cpp
           libamdhsacode/amd_hsa_code_util.cpp
           libamdhsacode/amd_hsa_locks.cpp
           libamdhsacode/amd_options.cpp
           libamdhsacode/amd_hsa_code.cpp
           lttng/rocm_hsa_tp.cpp
           lttng/rocm_trace_init.cpp)

if(UNIX)
  set(SRC_OS core/util/lnx/os_linux.cpp
	     libamdhsacode/lnx/amd_core_dump.cpp)
  set(SRC_XDNA core/driver/xdna/amd_xdna_driver.cpp
               core/runtime/amd_aie_agent.cpp
               core/runtime/amd_aie_aql_queue.cpp)
else()
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE NOMINMAX)
  set(SRC_OS core/util/win/os_win.cpp
	     libamdhsacode/win32/amd_core_dump.cpp)
endif()

if ( BUILD_THUNK_VIRTIO )
  list(APPEND SRCS core/driver/virtio/amd_kfd_virtio_driver.cpp)
  target_compile_definitions(hsa-runtime64 PRIVATE HSAKMT_VIRTIO_ENABLED=1)
endif()

target_sources( ${CORE_RUNTIME_TARGET} PRIVATE ${SRCS} ${SRC_OS} ${SRC_XDNA} )

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

if (${AMDCUID_FOUND})
  find_package(OpenSSL REQUIRED) # Required by CUID library
  target_include_directories(${CORE_RUNTIME_TARGET} PRIVATE ${AMDCUID_INCLUDE_DIR})
  target_link_libraries(${CORE_RUNTIME_TARGET} PRIVATE ${AMDCUID_LIB} OpenSSL::Crypto)
  set_target_properties(${CORE_RUNTIME_TARGET} PROPERTIES
              BUILD_RPATH "${AMDCUID_ROOT}/lib"
              INSTALL_RPATH "${AMDCUID_ROOT}/lib")
  target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HSA_ENABLE_AMDCUID_SUPPORT)
endif()

if (NOT DEFINED IMAGE_SUPPORT AND CMAKE_SYSTEM_PROCESSOR MATCHES "i?86|x86_64|amd64|AMD64|loongarch64")
  set (IMAGE_SUPPORT ON)
endif()
set (IMAGE_SUPPORT ${IMAGE_SUPPORT} CACHE BOOL "Build with image support (default: ON for x86, OFF elsewise).")

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

if (UNIX)
  target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE elf::elf dl pthread rt )
else()
  target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE Ws2_32 )
  target_link_directories(${CORE_RUNTIME_TARGET} PRIVATE ${DXCORE_LIB_PATH})
  target_link_libraries(${CORE_RUNTIME_TARGET} PRIVATE dxcore)
endif()

# For static package rocprofiler-register dependency is not required
# Link to hsakmt target for shared library builds
# Link to hsakmt-staticdrm target for static library builds
if( BUILD_SHARED_LIBS )
  if (UNIX)
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

    ## LTTng-UST tracepoint provider — see runtime/hsa-runtime/lttng/.
    ## Producer-side instrumentation that runs in parallel with the
    ## rocprofiler-register dispatch table; the dispatch table itself is
    ## unmodified.
    option(ROCR_ENABLE_LTTNG_UST "Build with LTTng-UST tracepoint emission" ON)
    if(ROCR_ENABLE_LTTNG_UST)
      # Vendored LTTng-UST: built from the lttng-ust + userspace-rcu git
      # submodules under shared/lttng/ (canonical, shared with CLR),
      # installed into the build tree at <build>/_deps/lttng-prefix/, and
      # shipped at runtime under /opt/rocm/lib/ alongside libhsa-runtime64.so.
      # libhsa-runtime64 picks them up via RPATH $ORIGIN (set below).
      # See shared/lttng/README.md.
      set(LTTNG_VENDORED_ROOT
          "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/lttng"
          CACHE PATH "Path to shared vendored LTTng tree (shared/lttng/)")
      if(NOT EXISTS "${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake")
        message(FATAL_ERROR
            "ROCR_ENABLE_LTTNG_UST=ON but shared/lttng/ was not found at\n"
            "    ${LTTNG_VENDORED_ROOT}\n"
            "Either checkout the rocm-systems super-repo (which provides "
            "shared/lttng/), set -DLTTNG_VENDORED_ROOT=<path>, or disable "
            "with -DROCR_ENABLE_LTTNG_UST=OFF.")
      endif()
      set(LTTNG_VENDORED_URCU_SRC "${LTTNG_VENDORED_ROOT}/userspace-rcu")
      set(LTTNG_VENDORED_UST_SRC  "${LTTNG_VENDORED_ROOT}/lttng-ust")
      include(${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake)

      find_package(PkgConfig REQUIRED)
      # Prepend the vendored .pc dir so pkg_check_modules picks it
      # before any system lttng-ust the distro might also ship.
      set(ENV{PKG_CONFIG_PATH} "${LTTNG_VENDORED_PKGCONFIG}:$ENV{PKG_CONFIG_PATH}")
      pkg_check_modules(LTTNG_UST REQUIRED IMPORTED_TARGET lttng-ust>=2.13)

      target_compile_definitions(${CORE_RUNTIME_TARGET} PRIVATE HSA_ENABLE_LTTNG_UST=1)
      target_include_directories(${CORE_RUNTIME_TARGET} PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/lttng
          ${LTTNG_VENDORED_INCLUDE_DIR})
      # Explicitly add the vendored lib directory to the link search path.
      # pkg_check_modules(IMPORTED_TARGET) generated PkgConfig::LTTNG_UST from
      # the stub .pc file written at configure time; in some pkg-config / CMake
      # combinations the -L directive in the stub is dropped from the imported
      # target's interface (the .so files do not exist yet at configure time
      # because the ExternalProject builds them later), leaving only -llttng-ust
      # / -llttng-ust-common in the link line and the linker fails with
      # "cannot find -llttng-ust". Adding the directory directly guarantees
      # the search path is present at link time.
      target_link_directories(${CORE_RUNTIME_TARGET} PRIVATE ${LTTNG_VENDORED_LIB_DIR})
      target_link_libraries(${CORE_RUNTIME_TARGET} PRIVATE
          PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})
      set(HSA_DEP_LTTNG_UST ON CACHE INTERNAL "")

      # Build the runtime only after the vendored libs exist.
      add_dependencies(${CORE_RUNTIME_TARGET} lttng_ust_vendored)

      # RPATH:
      #   BUILD_RPATH   — let in-tree tests find the vendored .so files.
      #   INSTALL_RPATH — at /opt/rocm/lib/libhsa-runtime64.so, the
      #                   vendored libs sit directly in /opt/rocm/lib/.
      # We append rather than overwrite because rocr-runtime sets RPATH
      # in several places; use APPEND to preserve any prior settings.
      set_property(TARGET ${CORE_RUNTIME_TARGET} APPEND PROPERTY
                   BUILD_RPATH "${LTTNG_VENDORED_LIB_DIR}")
      set_property(TARGET ${CORE_RUNTIME_TARGET} APPEND PROPERTY
                   INSTALL_RPATH "\$ORIGIN")

      # Stage the vendored .so files into the same build-tree directory
      # as libhsa-runtime64.so. This is what fixes downstream link
      # failures (e.g. TheRock CI: rocFFT, rocRoller, rocprofiler-systems
      # link against -lhsa-runtime64; the linker reads NEEDED entries
      # like liblttng-ust.so.1 and resolves them via the same -L path
      # used to find libhsa-runtime64.so). With the vendored .so files
      # co-located in the build dir, no extra -rpath-link or downstream
      # coordination is required. Mirrors the installed flat layout.
      #
      # PRE_LINK runs after object compilation but before the link step,
      # so the vendored .so files are present when the linker resolves
      # -llttng-ust via the target's link directory. Attaching to the
      # core runtime target directly (rather than using
      # add_custom_target + add_dependencies) avoids the cyclic
      # dependency that arises when $<TARGET_FILE_DIR:...> is used in a
      # separate custom target.
      if(NOT WIN32)
        # Use a sane shell-based existence check rather than
        # `cp ... 2>/dev/null || true`. The previous form silently
        # masked the case where the vendored .so files were missing
        # (e.g. the FetchContent build failed in a hard-to-spot way),
        # which then broke downstream link consumers later. Now we
        # explicitly probe for liblttng-ust.so* (the canonical
        # vendored library) and fail the build with an actionable
        # error message if absent. The cp itself uses the original
        # glob style and lets `set -e` surface any real failures.
        #
        # Single-line shell script: make's recipe parser does not
        # allow bare newlines inside a command argument.
        add_custom_command(TARGET ${CORE_RUNTIME_TARGET} PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_FILE_DIR:${CORE_RUNTIME_TARGET}>"
            COMMAND sh -c "set -e; if ! ls '${LTTNG_VENDORED_LIB_DIR}'/liblttng-ust.so* >/dev/null 2>&1; then echo 'error: no vendored LTTng-UST shared libraries found to stage from ${LTTNG_VENDORED_LIB_DIR}' >&2; exit 1; fi; cp -P '${LTTNG_VENDORED_LIB_DIR}'/liblttng*.so* '${LTTNG_VENDORED_LIB_DIR}'/liburcu*.so* '$<TARGET_FILE_DIR:${CORE_RUNTIME_TARGET}>/'"
            COMMENT "Staging vendored LTTng-UST .so files into ${CORE_RUNTIME_TARGET} build dir"
            VERBATIM)
      endif()

      # Install the vendored .so files directly under ${CMAKE_INSTALL_LIBDIR}
      # alongside libhsa-runtime64.so (no lttng/ subdirectory). This lets
      # downstream consumers that link -lhsa-runtime64 also resolve the
      # NEEDED liblttng-ust.so.1 etc. via the same -L path. The version-
      # suffixed symlinks are preserved so NEEDED entries that reference
      # e.g. liblttng-ust.so.1 resolve correctly. NB: when both HIP and
      # HSA are installed to the same prefix this writes the same files
      # twice; CMake's install rule is idempotent so that is harmless.
      install(DIRECTORY ${LTTNG_VENDORED_LIB_DIR}/
              DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT binary
              USE_SOURCE_PERMISSIONS
              FILES_MATCHING
              PATTERN "liblttng*.so*"
              PATTERN "liburcu*.so*")

      ## Symbol-coverage gate: post-build, diff the .so's exported HSA symbols
      ## against the migration inventory. Fail the build on any miss.
      add_custom_command(TARGET ${CORE_RUNTIME_TARGET} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E echo "Running LTTng symbol-coverage gate..."
          COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lttng_coverage_gate.sh
                  $<TARGET_FILE:${CORE_RUNTIME_TARGET}>
                  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lttng_migration_inventory.txt
          VERBATIM)
    else()
      set(HSA_DEP_LTTNG_UST OFF CACHE INTERNAL "")
    endif()
  else()
    target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt::hsakmt)
  endif()
else()
  if (UNIX)
    include_directories(${drm_INCLUDE_DIRS})
    target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt-staticdrm::hsakmt-staticdrm)
  else()
    target_link_libraries ( ${CORE_RUNTIME_TARGET} PRIVATE hsakmt-staticdrm::hsakmt-staticdrm)
  endif()
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
  if (UNIX)
    target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE -Wl,$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}/${LNKSCR}
      -Wl,--whole-archive $<TARGET_FILE:${CORE_RUNTIME_NAME}::${CORE_RUNTIME_TARGET}> -Wl,--no-whole-archive)
  else()
    # Windows: Link the static library directly without linker script
    target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE ${CORE_RUNTIME_TARGET})
  endif()
  add_dependencies( ${CORE_RUNTIME_NAME} ${CORE_RUNTIME_TARGET} )

  ## Add external link requirements.
  target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE hsakmt-staticdrm::hsakmt-staticdrm )
  if (UNIX)
    target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE elf::elf dl pthread rt )
  else()
    target_link_libraries ( ${CORE_RUNTIME_NAME} INTERFACE Ws2_32 )
  endif()
  install ( TARGETS ${CORE_RUNTIME_NAME} EXPORT ${CORE_RUNTIME_NAME}Targets )
endif()

## Set install information
# Installs binaries and exports the library usage data to ${HSAKMT_TARGET}Targets
if (NOT WIN32)
  install ( TARGETS ${CORE_RUNTIME_TARGET} EXPORT ${CORE_RUNTIME_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary )

  install ( TARGETS ${CORE_RUNTIME_TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )
endif()

# Install license
if(ENABLE_ASAN_PACKAGING)
  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan )
endif()
if (NOT WIN32)
  install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary )
endif()

# Install public headers
install ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hsa COMPONENT dev )

## Configure and install package config file
# Record our usage data for clients find_package calls.
if (NOT WIN32)
  install ( EXPORT ${CORE_RUNTIME_NAME}Targets
    FILE ${CORE_RUNTIME_NAME}Targets.cmake
    NAMESPACE ${CORE_RUNTIME_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
    COMPONENT dev)
endif()

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

if (NOT WIN32)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CORE_RUNTIME_NAME}-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${CORE_RUNTIME_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
    COMPONENT dev)
endif()

# Install build files needed only when using a static build.
if( NOT ${BUILD_SHARED_LIBS} )
  if (UNIX)
    # libelf find package module
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindLibElf.cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/COPYING-CMAKE-SCRIPTS
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
      COMPONENT dev)
    # Linker script (defines function aliases)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${LNKSCR}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CORE_RUNTIME_NAME}
      COMPONENT dev)
  endif()
endif()
