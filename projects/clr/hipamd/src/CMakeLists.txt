# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)

set(VERSION_MAJOR_AMDHIP ${HIP_VERSION_MAJOR})
set(VERSION_MINOR_AMDHIP ${HIP_VERSION_MINOR})

if(NOT DEFINED THEROCK_SANITIZER)
  set(THEROCK_SANITIZER)
endif()

if(ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  message(STATUS "Building with sanitizer")
  set(ASAN_LINKER_FLAGS "-fsanitize=address")
  set(ASAN_COMPILER_FLAGS "-fno-omit-frame-pointer -fsanitize=address")

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(BUILD_SHARED_LIBS)
      set(ASAN_COMPILER_FLAGS "${ASAN_COMPILER_FLAGS} -shared-libsan")
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -shared-libsan")
    else()
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -static-libsan")
    endif()
  endif()

  # If its being built by TheRock, CMAKE_<LANG>_FLAGS should have the sanitizer flags
  # If not, we append it.
  if(ADDRESS_SANITIZER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LINKER_FLAGS} -s -Wl,--build-id=sha1")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINKER_FLAGS} -Wl,--build-id=sha1")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
endif()

option(BUILD_SHARED_LIBS "Build the shared library" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(BUILD_SHARED_LIBS)
  add_library(amdhip64 SHARED)
  if(WIN32)
    set_target_properties(amdhip64 PROPERTIES RUNTIME_OUTPUT_NAME "amdhip64_${HIP_VERSION_MAJOR}")
  endif()
  # Windows doesn't have a strip utility, so CMAKE_STRIP won't be set.
  if((CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT ("${CMAKE_STRIP}" STREQUAL ""))
    add_custom_command(TARGET amdhip64 POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:amdhip64>)
  endif()
else()
  add_library(amdhip64 STATIC $<TARGET_OBJECTS:rocclr>)
endif()

set_target_properties(amdhip64 PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  # Workaround for many places in the HIP project
  # having hardcoded references to build/lib/libamdhip64.so
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if(WIN32)
  set_target_properties(amdhip64 PROPERTIES CXX_STANDARD 20)
else()
  set_target_properties(amdhip64 PROPERTIES CXX_STANDARD 17)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set_target_properties(amdhip64 PROPERTIES OUTPUT_NAME "amdhip64")
else()
  set_target_properties(amdhip64 PROPERTIES OUTPUT_NAME "amdhip32")
endif()

# Disable versioning for Windows
# as currently HIP_LIB_VERSION_STRING and HIP_LIB_VERSION_MAJOR
# are not being populated
if(NOT WIN32)
  if(BUILD_SHARED_LIBS)
    set_target_properties(amdhip64 PROPERTIES
      VERSION ${HIP_LIB_VERSION_STRING}
      SOVERSION ${HIP_LIB_VERSION_MAJOR})
  endif()
endif()

target_sources(amdhip64 PRIVATE
  hip_activity.cpp
  hip_code_object.cpp
  hip_context.cpp
  hip_device_runtime.cpp
  hip_device.cpp
  hip_error.cpp
  hip_event.cpp
  hip_event_ipc.cpp
  hip_fatbin.cpp
  hip_global.cpp
  hip_executionctx.cpp
  hip_graph_internal.cpp
  hip_graph.cpp
  hip_hmm.cpp
  hip_intercept.cpp
  hip_log.cpp
  hip_memory.cpp
  hip_mempool.cpp
  hip_mempool_impl.cpp
  hip_module.cpp
  hip_peer.cpp
  hip_platform.cpp
  hip_profile.cpp
  hip_storage.cpp
  hip_stream_ops.cpp
  hip_stream.cpp
  hip_surface.cpp
  hip_texture.cpp
  hip_gl.cpp
  hip_vm.cpp
  hip_api_trace.cpp
  hip_table_interface.cpp
  hip_table_interface_c.cpp
  hip_comgr_helper.cpp
  hip_library.cpp
  profiler/hip_clr_profiler.cpp
  profiler/hip_clr_dispatch_wrappers.cpp)

if(WIN32)
  target_sources(amdhip64 PRIVATE hip_runtime.cpp)
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32)
    set(AMDHIP_DEF_OUTPUT "${PROJECT_BINARY_DIR}/amdhip.def")

    # ROCCLR_ENABLE_HSA is known at configure time; use configure_file to
    # conditionally include the HSA export block instead of invoking the C
    # preprocessor at build time (cl.exe does not support GCC-style -E -P -o).
    if(ROCCLR_ENABLE_HSA)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/hsa_exports.def")
      file(READ "${CMAKE_CURRENT_LIST_DIR}/hsa_exports.def" HSA_EXPORTS)
    else()
      set(HSA_EXPORTS "")
    endif()

    configure_file("${CMAKE_CURRENT_LIST_DIR}/amdhip.def.in" "${AMDHIP_DEF_OUTPUT}" @ONLY)

    # For MSVC, pass the .def file directly to the linker with /DEF: flag
    string(REPLACE "/" "\\" AMDHIP_DEF_OUTPUT_NATIVE "${AMDHIP_DEF_OUTPUT}")
    target_link_options(amdhip64 PRIVATE "/DEF:${AMDHIP_DEF_OUTPUT_NATIVE}")
    set_target_properties(amdhip64 PROPERTIES LINK_DEPENDS "${AMDHIP_DEF_OUTPUT}")
  else()
    target_link_libraries(amdhip64 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hip_hcc.map.in")
    set_target_properties(amdhip64 PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hip_hcc.map.in")
  endif()
endif()

if(WIN32)
  configure_file(hip_hcc_in.rc.in hip_hcc_info.rc @ONLY)
  target_sources(amdhip64 PRIVATE hip_hcc_info.rc)
endif()

target_include_directories(amdhip64
  PRIVATE
    ${HIP_COMMON_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)

target_compile_definitions(amdhip64 PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(amdhip64 PRIVATE ${OPENGL_LIBRARIES})
target_link_libraries(amdhip64 PRIVATE ${CMAKE_DL_LIBS})
# Add link to comgr, hsa-runtime and other required libraries in target files
# This is required for static libraries
if(NOT BUILD_SHARED_LIBS)
  find_package(hsa-runtime64)
  find_package(amd_comgr)
  target_link_libraries(amdhip64 PRIVATE
    pthread numa rt c amd_comgr hsa-runtime64::hsa-runtime64)
endif()

# Note in static case we cannot link against rocclr.
# If we would, we'd also have to export rocclr and have hipcc pass it to the linker.
if(BUILD_SHARED_LIBS)
  target_link_libraries(amdhip64 PRIVATE rocclr)
else()
  target_compile_definitions(amdhip64 PRIVATE $<TARGET_PROPERTY:rocclr,COMPILE_DEFINITIONS>)
  target_include_directories(amdhip64 PRIVATE $<TARGET_PROPERTY:rocclr,INCLUDE_DIRECTORIES>)
endif()

# Optional rocm-kpack support for kpack split artifacts
# This option will be enabled permanently at a future point and gates the use
# of the rocm-kpack library for detecting ROCm multi-arch archives in
# distributions of ROCm so that the CLR can load them just as it would normal
# fat binaries. See the WIP repo: https://github.com/ROCm/rocm-kpack (which
# will be migrated to rocm-systems when ready).
option(ROCM_KPACK_ENABLED "Enable kpack runtime loading for split device code" OFF)
if(ROCM_KPACK_ENABLED)
  find_package(rocm-kpack REQUIRED)
  target_compile_definitions(amdhip64 PRIVATE ROCM_KPACK_ENABLED=1)
  target_link_libraries(amdhip64 PRIVATE rocm::rocm_kpack)
endif()

# Short-Term solution for pre-compiled headers for online compilation
# Enable pre compiled header
if(__HIP_ENABLE_PCH)
  find_package(LLVM REQUIRED CONFIG
    HINTS
      ${ROCM_PATH}/llvm)
  # find_package(LLVM) returns the lib/cmake/llvm location. We require the root.
  if(NOT DEFINED HIP_LLVM_ROOT)
    set(HIP_LLVM_ROOT "${LLVM_DIR}/../../..")
    if(NOT EXISTS "${HIP_LLVM_ROOT}/bin/clang")
      # required because Rock installs llvm to /opt/rocm/lib/llvm
      set(HIP_LLVM_ROOT "${LLVM_DIR}/../../llvm")
    endif()
  endif()

  execute_process(COMMAND sh -c "${CMAKE_CURRENT_SOURCE_DIR}/hip_embed_pch.sh ${HIP_COMMON_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/include ${HIP_LLVM_ROOT}" COMMAND_ECHO STDERR RESULT_VARIABLE EMBED_PCH_RC WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  if (EMBED_PCH_RC AND NOT EMBED_PCH_RC EQUAL 0)
    message(FATAL_ERROR "Failed to embed PCH")
  endif()

  target_compile_definitions(amdhip64 PRIVATE __HIP_ENABLE_PCH)
  target_sources(amdhip64 PRIVATE ${CMAKE_BINARY_DIR}/hip_pch.o)
endif()

# Add hiprtc
add_subdirectory(hiprtc)

# HRR in-tree capture layer
add_subdirectory(hrr)

#############################
# Profiling API support
#############################
# Generate profiling API macros/structures header
option(USE_PROF_API "Enable roctracer integration" ON)
# Enable profiling API
if(USE_PROF_API)
  set(PROF_API_STR "${PROJECT_BINARY_DIR}/include/hip/amd_detail/hip_prof_str.h")
  set(PROF_API_STR_IN "${CMAKE_SOURCE_DIR}/hipamd/include/hip/amd_detail/hip_prof_str.h")
  set(PROF_API_HDR "${HIP_COMMON_INCLUDE_DIR}/hip/hip_runtime_api.h")
  set(PROF_GL_HDR "${CMAKE_SOURCE_DIR}/hipamd/include/hip/amd_detail/amd_hip_gl_interop.h")
  set(PROF_API_DEPRECATED "${HIP_COMMON_INCLUDE_DIR}/hip/hip_deprecated.h")
  set(PROF_API_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
  set(PROF_API_GEN "${CMAKE_CURRENT_SOURCE_DIR}/hip_prof_gen.py")
  set(PROF_API_LOG "${PROJECT_BINARY_DIR}/hip_prof_gen.log.txt")
  set(PROF_API_NEWHDR "${PROJECT_BINARY_DIR}/new_header.h")

  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import CppHeaderParser"
                  RESULT_VARIABLE CPP_HEADER_PARSER
                  OUTPUT_QUIET)

  if(NOT ${CPP_HEADER_PARSER} EQUAL 0)
    message(FATAL_ERROR "\
    The \"CppHeaderParser\" Python3 package is not installed. \
    Please install it using the following command: \"pip3 install CppHeaderParser\".\
  ")
  endif()

  add_custom_command(OUTPUT ${PROF_API_NEWHDR}.i
    COMMAND ${CMAKE_COMMAND} -E cat ${PROF_API_HDR} ${PROF_GL_HDR} > ${PROF_API_NEWHDR}
    COMMAND ${CMAKE_C_COMPILER}
        "-D$<JOIN:$<TARGET_PROPERTY:amdhip64,COMPILE_DEFINITIONS>,;-D>"
        "-I$<JOIN:$<TARGET_PROPERTY:amdhip64,INCLUDE_DIRECTORIES>,;-I>"
        "-DHIP_INCLUDE_HIP_HIP_RUNTIME_PT_API_H=1"
        ${c_flags}
        $<TARGET_PROPERTY:amdhip64,COMPILE_OPTIONS>
        ${CPP_EXTRA_C_FLAGS}
        -E ${PROF_API_NEWHDR} -o ${PROF_API_NEWHDR}.i
    COMMAND_EXPAND_LISTS VERBATIM
    IMPLICIT_DEPENDS C ${PROF_API_HDR} ${PROF_GL_HDR} ${PROF_API_DEPRECATED}
    DEPENDS ${PROF_API_HDR} ${PROF_GL_HDR} ${PROF_API_DEPRECATED}
    COMMENT "Generating new header from hip_runtime_api.h")

  add_custom_command(OUTPUT ${PROF_API_STR}
    COMMAND ${Python3_EXECUTABLE} ${PROF_API_GEN} -v -t --priv ${PROF_API_NEWHDR}.i ${PROF_API_SRC} ${PROF_API_STR_IN} ${PROF_API_STR}
    DEPENDS ${PROF_API_STR_IN} ${PROF_API_NEWHDR}.i ${PROF_API_GEN}
    COMMENT "Generating profiling primitives: ${PROF_API_STR}")

  add_custom_target(gen-prof-api-str-header ALL
    DEPENDS ${PROF_API_STR}
    SOURCES ${PROF_API_NEWHDR}.i)

  set_target_properties(amdhip64 PROPERTIES PUBLIC_HEADER ${PROF_API_STR})

  find_path(PROF_API_HEADER_DIR prof_protocol.h
    HINTS
    ${PROF_API_HEADER_PATH}
    PATHS
    ${ROCM_PATH}/roctracer
    PATH_SUFFIXES
    include/ext)

  if(NOT PROF_API_HEADER_DIR)
    message(WARNING "Profiling API header not found. Disabling roctracer integration. Use -DPROF_API_HEADER_PATH=<path to prof_protocol.h header>")
  else()
    target_include_directories(amdhip64 PUBLIC ${PROF_API_HEADER_DIR})
    message(STATUS "Profiling API: ${PROF_API_HEADER_DIR}")
  endif()

  add_dependencies(amdhip64 gen-prof-api-str-header)
endif()

target_compile_definitions(amdhip64 PUBLIC USE_PROF_API=1)

if(WIN32 OR NOT BUILD_SHARED_LIBS)
    # rocprofiler-register is not support on Windows
    set(HIP_ENABLE_ROCPROFILER_REGISTER OFF)
else()
  option(HIP_ENABLE_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
endif()

if(HIP_ENABLE_ROCPROFILER_REGISTER)
    find_package(rocprofiler-register REQUIRED
        HINTS $ENV{rocprofiler_register_ROOT} $ENV{ROCPROFILER_REGISTER_ROOT} ${CMAKE_INSTALL_PREFIX}
        PATHS ${ROCM_PATH})

    # don't use HIP_VERSION_PATCH because it is too large (> 100) for rocprofiler register
    target_compile_definitions(amdhip64 PRIVATE HIP_ROCPROFILER_REGISTER=1
        HIP_ROCP_REG_VERSION_MAJOR=${HIP_VERSION_MAJOR}
        HIP_ROCP_REG_VERSION_MINOR=${HIP_VERSION_MINOR}
        HIP_ROCP_REG_VERSION_PATCH=0)
    target_link_libraries(amdhip64 PRIVATE rocprofiler-register::rocprofiler-register)
    set_target_properties(amdhip64 PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

# LTTng-UST tracepoint emission. ON by default; OFF is supported only as
# an emergency escape valve. The packaging declarations in
# hipamd/packaging/CMakeLists.txt are gated on this same option so the
# binary's NEEDED entry is always package-resolvable.
if(WIN32 OR NOT BUILD_SHARED_LIBS)
    set(HIP_ENABLE_LTTNG_UST OFF)
else()
    option(HIP_ENABLE_LTTNG_UST "Build with LTTng-UST tracepoint emission" ON)
endif()

# LTTng-UST link mode for amdhip64:
#   dynamic (default) - amdhip64 does NOT link liblttng-ust (zero
#     DT_NEEDED on lttng-ust/urcu). Tracepoint call sites remain
#     compiled in; probe registration/emission only activates when the
#     separate rocm_hip_lttng_provider DSO is loaded at runtime (e.g.
#     LD_PRELOAD). This is LTTng-UST's documented "dynamic loading"
#     pattern -- see shared/lttng/lttng-ust/README.md.
#   direct - legacy behavior: amdhip64 links liblttng-ust directly and
#     creates the probes itself, no separate provider DSO is built. Kept
#     only for easy A/B comparison during development of the dynamic
#     mode; not intended for production use.
set(HIP_LTTNG_UST_LINK_MODE "dynamic" CACHE STRING
    "LTTng-UST link mode for amdhip64: 'dynamic' (default, no DT_NEEDED \
on lttng-ust; opt-in via LD_PRELOAD of rocm_hip_lttng_provider.so) or \
'direct' (legacy fallback, amdhip64 links liblttng-ust directly)")
set_property(CACHE HIP_LTTNG_UST_LINK_MODE PROPERTY STRINGS dynamic direct)
if(NOT HIP_LTTNG_UST_LINK_MODE STREQUAL "dynamic" AND NOT HIP_LTTNG_UST_LINK_MODE STREQUAL "direct")
    message(FATAL_ERROR
        "HIP_LTTNG_UST_LINK_MODE must be 'dynamic' or 'direct' "
        "(got '${HIP_LTTNG_UST_LINK_MODE}')")
endif()

if(HIP_ENABLE_LTTNG_UST)
    # Vendored LTTng-UST: built from the lttng-ust + userspace-rcu git
    # submodules under shared/lttng/ (canonical, shared with rocr-runtime),
    # installed into the build tree at <build>/_deps/lttng-prefix/, and
    # shipped at runtime under /opt/rocm/lib/ alongside libamdhip64.so
    # (libamdhip64 picks them up via RPATH $ORIGIN — see
    # set_target_properties below). See shared/lttng/README.md.
    set(LTTNG_VENDORED_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../../../../shared/lttng"
        CACHE PATH "Path to shared vendored LTTng tree (shared/lttng/)")
    if(NOT EXISTS "${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake")
        message(FATAL_ERROR
            "HIP_ENABLE_LTTNG_UST=ON but shared/lttng/ was not found at\n"
            "    ${LTTNG_VENDORED_ROOT}\n"
            "Either checkout the rocm-systems super-repo (which provides "
            "shared/lttng/), set -DLTTNG_VENDORED_ROOT=<path>, or disable "
            "with -DHIP_ENABLE_LTTNG_UST=OFF.")
    endif()
    set(LTTNG_VENDORED_URCU_SRC "${LTTNG_VENDORED_ROOT}/userspace-rcu")
    set(LTTNG_VENDORED_UST_SRC  "${LTTNG_VENDORED_ROOT}/lttng-ust")
    include(${LTTNG_VENDORED_ROOT}/cmake/VendorLttng.cmake)

    find_package(PkgConfig REQUIRED)
    # Prepend the vendored .pc dir so pkg_check_modules picks it before
    # any system lttng-ust the distro might also ship.
    set(ENV{PKG_CONFIG_PATH} "${LTTNG_VENDORED_PKGCONFIG}:$ENV{PKG_CONFIG_PATH}")
    pkg_check_modules(LTTNG_UST REQUIRED IMPORTED_TARGET lttng-ust>=2.13)

    # HIP_ENABLE_LTTNG_UST + the provider header/include path are needed by
    # amdhip64 in BOTH link modes: the call sites (via rocm_trace_emit.h)
    # and the definition-only TU (lttng/rocm_hip_tp.cpp) both include
    # rocm_hip_tp.h regardless of dynamic vs direct linkage.
    target_compile_definitions(amdhip64 PRIVATE HIP_ENABLE_LTTNG_UST=1)
    target_include_directories(amdhip64 PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/lttng
        ${LTTNG_VENDORED_INCLUDE_DIR})

    # lttng/rocm_hip_tp.cpp is the definition-only TU that actually
    # generates the tracepoint provider probes (LTTNG_UST_TRACEPOINT_
    # CREATE_PROBES / _DEFINE, in direct mode) or the dynamic-linkage
    # declaration (in dynamic mode); lttng/rocm_trace_init.cpp defines the
    # runtime-wide kill switch (rocm_hip_trace_g_disabled) that every
    # call site in rocm_trace_emit.h references, plus the dlmopen
    # SEGFAULT mitigation constructor. Without both compiled into
    # amdhip64, the library has undefined references to the tracepoint
    # symbols and rocm_hip_trace_g_disabled at link/load time.
    target_sources(amdhip64 PRIVATE
        lttng/rocm_hip_tp.cpp
        lttng/rocm_trace_init.cpp)

    # Build amdhip64 only after the vendored libs exist (the .pc query
    # above happens at configure time, but the actual .so files are only
    # produced when the lttng_ust_vendored ExternalProject runs at build
    # time; without this dependency, the link step -- or, in dynamic
    # mode, the provider DSO's link step -- races the build).
    add_dependencies(amdhip64 lttng_ust_vendored)

    if(HIP_LTTNG_UST_LINK_MODE STREQUAL "dynamic")
        # Dynamic mode: lttng/rocm_hip_tp.cpp must NOT create the probes
        # itself (see the mode branch inside that file); only DEFINE +
        # PROBE_DYNAMIC_LINKAGE. amdhip64 does not link PkgConfig::LTTNG_UST,
        # does not get an LTTng RPATH entry, and does not stage the vendored
        # .so files -- none of that is needed since amdhip64 has zero
        # DT_NEEDED on lttng-ust/urcu in this mode.
        set_source_files_properties(lttng/rocm_hip_tp.cpp PROPERTIES
            COMPILE_DEFINITIONS "HIP_LTTNG_UST_DYNAMIC_LINKAGE=1")

        # Separate tracepoint-provider DSO: defines LTTNG_UST_TRACEPOINT_
        # CREATE_PROBES only and links liblttng-ust directly. Loaded at
        # runtime (e.g. LD_PRELOAD) to activate rocm_hip:* tracepoint
        # emission -- LTTng-UST's documented "dynamic loading" pattern
        # (see shared/lttng/lttng-ust/README.md and
        # lttng/rocm_hip_tp_provider.c).
        #
        # BUILD-TREE NOTE: built into the same output directory as amdhip64
        # for build-tree simplicity (colocated build artifacts); this is
        # not the installed layout. Packaging (hipamd/packaging/CMakeLists.txt)
        # installs this DSO into a dedicated ${CMAKE_INSTALL_LIBDIR}/rocm-lttng/
        # subdirectory instead of flat beside libamdhip64.so.
        add_library(rocm_hip_lttng_provider SHARED
            lttng/rocm_hip_tp_provider.c)
        set_target_properties(rocm_hip_lttng_provider PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
            POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(rocm_hip_lttng_provider PRIVATE HIP_ENABLE_LTTNG_UST=1)
        target_include_directories(rocm_hip_lttng_provider PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lttng
            ${LTTNG_VENDORED_INCLUDE_DIR})
        target_link_directories(rocm_hip_lttng_provider PRIVATE ${LTTNG_VENDORED_LIB_DIR})
        # --no-as-needed: this TU never calls an lttng-ust API directly (it
        # only expands the CREATE_PROBES registration-data macros), so a
        # linker configured with --as-needed by default would drop the
        # DT_NEEDED entry entirely, defeating the purpose of this DSO.
        # Mirrors lttng-ust's own doc/examples/demo Makefile.
        target_link_libraries(rocm_hip_lttng_provider PRIVATE
            "-Wl,--no-as-needed" PkgConfig::LTTNG_UST)
        add_dependencies(rocm_hip_lttng_provider lttng_ust_vendored)
        set_target_properties(rocm_hip_lttng_provider PROPERTIES
            BUILD_RPATH   "${LTTNG_VENDORED_LIB_DIR}"
            INSTALL_RPATH "\$ORIGIN")
    else()
        # direct mode: preserve pre-split behavior exactly (amdhip64 both
        # defines and creates the probes via lttng/rocm_hip_tp.cpp, and
        # links liblttng-ust itself). Kept only for A/B comparison during
        # development; not intended for production use.
        target_link_directories(amdhip64 PRIVATE ${LTTNG_VENDORED_LIB_DIR})
        target_link_libraries(amdhip64 PRIVATE PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})

        # RPATH:
        #   BUILD_RPATH   — let in-tree tests find the vendored .so files at
        #                   <build>/_deps/lttng-prefix/lib/.
        #   INSTALL_RPATH — at /opt/rocm/lib/libamdhip64.so, the vendored
        #                   libs sit directly in /opt/rocm/lib/ (no lttng/
        #                   subdir). The literal $ORIGIN expands to
        #                   /opt/rocm/lib at runtime.
        set_target_properties(amdhip64 PROPERTIES
            BUILD_RPATH   "${LTTNG_VENDORED_LIB_DIR}"
            INSTALL_RPATH "\$ORIGIN")

        # Stage the vendored .so files into the same build-tree directory as
        # libamdhip64.so. This is what fixes downstream link failures (e.g.
        # TheRock CI: rocFFT, rocRoller, rocprofiler-systems link against
        # -lamdhip64 / -lhsa-runtime64; the linker reads NEEDED entries like
        # liblttng-ust.so.1 and resolves them via the same -L path used to
        # find libamdhip64.so). With the vendored .so files co-located in
        # the build dir, no extra -rpath-link or downstream coordination is
        # required. Mirrors the installed flat layout under /opt/rocm/lib/.
        #
        # PRE_LINK runs after object compilation but before the link step,
        # so the vendored .so files are present when the linker resolves
        # -llttng-ust via the target's link directory. Attaching to the
        # amdhip64 target directly (rather than using add_custom_target +
        # add_dependencies) avoids the cyclic dependency that arises when
        # $<TARGET_FILE_DIR:amdhip64> is used in a separate custom target.
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
            add_custom_command(TARGET amdhip64 PRE_LINK
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "$<TARGET_FILE_DIR:amdhip64>"
                COMMAND sh -c "set -e; if ! ls '${LTTNG_VENDORED_LIB_DIR}'/liblttng-ust.so* >/dev/null 2>&1; then echo 'error: no vendored LTTng-UST shared libraries found to stage from ${LTTNG_VENDORED_LIB_DIR}' >&2; exit 1; fi; cp -P '${LTTNG_VENDORED_LIB_DIR}'/liblttng*.so* '${LTTNG_VENDORED_LIB_DIR}'/liburcu*.so* '$<TARGET_FILE_DIR:amdhip64>/'"
                COMMENT "Staging vendored LTTng-UST .so files into amdhip64 build dir"
                VERBATIM)
        endif()
    endif()

    # NB: install rules for the vendored .so files (and, in dynamic mode,
    # the rocm_hip_lttng_provider DSO) live in
    # hipamd/packaging/CMakeLists.txt under COMPONENT binary, gated on
    # HIP_ENABLE_LTTNG_UST and HIP_LTTNG_UST_LINK_MODE.

    # Per-packet AQL kernel-dispatch tracepoint lives at the HIP CLR packet-
    # write site (rocclr/device/rocm/rocvirtual.cpp). rocclr is built as a
    # static archive and linked into amdhip64; the tracepoint emit symbols
    # live in amdhip64 (via lttng/rocm_hip_tp.cpp), but rocvirtual.cpp needs
    # the include path, the HIP_ENABLE_LTTNG_UST macro, and the LTTng-UST
    # headers to compile the call sites. Apply all to the rocclr target here.
    #
    # rocm_trace_emit_curated.h transitively #include <hip/hip_runtime_api.h>
    # for the HIP types in the per-API typed-emit signatures, which in turn
    # pulls in <hip/hip_version.h> (generated at configure time into
    # ${PROJECT_BINARY_DIR}/include/hip/) and other headers from
    # ${PROJECT_SOURCE_DIR}/include. Mirror the three include directories
    # amdhip64 itself uses (see the target_include_directories above), so
    # rocclr — which is built as a static archive and linked into amdhip64
    # but does not normally inherit the HIP include wiring — can compile the
    # call sites in device/rocm/rocvirtual.cpp.
    if(TARGET rocclr)
        target_compile_definitions(rocclr PRIVATE HIP_ENABLE_LTTNG_UST=1)
        target_include_directories(rocclr PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lttng
            ${LTTNG_VENDORED_INCLUDE_DIR}
            ${LTTNG_UST_INCLUDE_DIRS}
            ${HIP_COMMON_INCLUDE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_BINARY_DIR}/include)
        add_dependencies(rocclr lttng_ust_vendored)
    endif()

    # Symbol-coverage gate: every exported HIP/__hip text symbol from the
    # built libamdhip64.so must appear in one of the checked-in migration
    # inventories. Fails the build on any miss.
    add_custom_command(TARGET amdhip64 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Running LTTng symbol-coverage gate..."
        COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_coverage_gate.sh
                $<TARGET_FILE:amdhip64>
                ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_migration_inventory.txt
                ${CMAKE_CURRENT_LIST_DIR}/../scripts/lttng_migration_inventory_c.txt
        VERBATIM)
endif()

add_custom_command(TARGET amdhip64 POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/share/hip/.hipInfo ${PROJECT_BINARY_DIR}/lib/.hipInfo)
add_custom_command(TARGET amdhip64 POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
add_custom_command(TARGET amdhip64 POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E copy_directory ${HIP_COMMON_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/include)

add_library(host INTERFACE)
target_link_libraries(host INTERFACE amdhip64)

add_library(device INTERFACE)
target_link_libraries(device INTERFACE host)

# Current packaging assumes that HIP runtime will always be installed in ${ROCM_PATH}/lib
# This is false to assume, because some distros like CentOS will use the lib64 directory instead of lib
# Relying on CMake to choose the library directory for us will default in that case to lib64
# Hence there will be a mismatch between where HIP is installed and where CMake thinks it is

INSTALL(TARGETS amdhip64 host device
  EXPORT hip-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(EXPORT hip-targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip NAMESPACE hip::)

INSTALL(TARGETS amdhip64 host device
  EXPORT hip-lang-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
INSTALL(EXPORT hip-lang-targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang NAMESPACE hip-lang::)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${HIP_COMMON_DIR}/hip-lang-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config.cmake
  INSTALL_DESTINATION ${CONFIG_LANG_PACKAGE_INSTALL_DIR}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config-version.cmake
  VERSION "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}"
  COMPATIBILITY SameMajorVersion)
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hip-lang-config-version.cmake
    DESTINATION
    ${CONFIG_LANG_PACKAGE_INSTALL_DIR}/
    )
