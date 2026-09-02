cmake_minimum_required(VERSION 3.16)

set(RCCL_MICRO_TEST_SOURCES
  p2p-test.cc
  rma-proxy-progress-test.cc          # RMA proxy inflight-request coverage
  group-test.cc
  devcomm-test.cc
  fakes/nccl_fakes.cc                  # reusable stubs for nccl* symbols
  fakes/p2p_fakes.cc                   # p2p-specific stubs (arch/topo + alloc emulators)
  fakes/rma_fakes.cc                   # rma_proxy_progress externals (circular-buf / destroy)
  fakes/comm_fakes.cc                  # comm-lifecycle seams (real in init.cc; faked here)
  fakes/recorder_fakes.cc              # src/recorder.cc
  fakes/utils_fakes.cc                 # src/misc/utils.cc (targets without the real utils.cc)
  fakes/collective_stubs.cc            # fail-loud collective launch + transport floor
  fakes/devcomm_fakes.cc
  fakes/hip_fakes.cc                   # HIP runtime seams + link-satisfying stubs
  ../common/main_altrsmi.cpp           # minimal gtest main, no RCCL deps
  ../common/ProcessIsolatedTestRunner.cpp
)

# =====================================================================
# In-RCCL-build integration.
#
# When this directory is pulled in via add_subdirectory(host) from
# test/CMakeLists.txt (i.e. as part of ./install.sh -t), BUILD_TESTS is
# set and the parent build has already established the toolchain,
# project(), and the shared RCCL_COMMON_INCLUDE_DIRS /
# RCCL_COMMON_COMPILE_DEFS / MICRO_TEST_LINK_LIBS variables. In that mode
# we only build the host-only micro-test binary(ies) here and return
# before the standalone project() setup below.
#
# When invoked directly (cmake -B build on this directory) BUILD_TESTS is
# not defined, so we fall through to the standalone project that builds
# rccl-HostUnitTests + rccl-UnitTestsMicro from a hipified snapshot.
# =====================================================================
if(BUILD_TESTS)
  # rccl-UnitTestsMicro: standalone unit-test binary compiled directly into
  # the test binary. This lets us use preprocessor and link-time substitution
  # to provide test doubles for anything that would require a GPU or other
  # special hardware. Deliberately NOT linking librccl.so or the HIP runtime,
  # so any un-shimmed HIP call surfaces as a link error rather than binding
  # libamdhip64.so (HIP compile requirements are applied below without linking).
  add_executable(rccl-UnitTestsMicro ${RCCL_MICRO_TEST_SOURCES})

  target_include_directories(rccl-UnitTestsMicro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/hipify/src/devcomm   # devcomm_v22902.h, included unqualified
    ${RCCL_COMMON_INCLUDE_DIRS})
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE ${RCCL_COMMON_COMPILE_DEFS})

  # Mirrors the opt-out in test/CMakeLists.txt, whose loop does not reach this target.
  # Inheriting ENABLE_ROCSHMEM_GIN defaults NCCL_GIN_ANVIL_SDMA_ENABLE to 1, which pulls
  # in rocSHMEM's sdma device headers.
  if(ENABLE_ROCSHMEM_GIN)
    target_compile_definitions(rccl-UnitTestsMicro PRIVATE NCCL_GIN_ANVIL_SDMA_ENABLE=0)
  endif()

  # Apply HIP's *compile* usage requirements (headers, device intrinsics/codegen,
  # compile defs) WITHOUT linking the HIP runtime. $<COMPILE_ONLY:hip::host> would
  # express this in one line but requires CMake >= 3.27; pulling the interface
  # properties directly keeps this valid at the project's declared minimum (3.16).
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
      target_compile_options(rccl-UnitTestsMicro PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
    endif()
  endforeach()

  target_link_libraries(rccl-UnitTestsMicro PRIVATE ${MICRO_TEST_LINK_LIBS})

  # Pass the absolute path of hipified source files through as string
  # literals so the #include in the test translation unit stays independent of
  # the build directory layout.
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE
    P2P_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/transport/p2p_tmp.cc"
    RMA_PROXY_PROGRESS_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/rma/rma_proxy_progress.cc"
    GROUP_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/group.cc"
    DEVCOMM_V22902_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/devcomm/devcomm_v22902.cc"
    DEVCOMM_V22907_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/devcomm/devcomm_v22907.cc"
  )

  # Make sure the hipify step has run before trying to compile the test.
  add_dependencies(rccl-UnitTestsMicro hipify_all copy_nccl_device_headers)

  # Always build this binary with llvm source-based coverage.
  # Host-only micro-test: all HIP/device symbols are satisfied by fakes/, so it
  # needs HIP host compile requirements but no amdgcn device codegen. Compile
  # host-only (like the standalone build below) so llvm coverage instrumentation
  # never reaches the device pass -- there it pulls in LLVM profiling runtime
  # symbols (__llvm_profile_raw_version, __start___llvm_prf_cnts, ...) that do
  # not exist on device, making amdgcn-link fail. --offload-host-only removes the
  # device pass entirely, which is robust across clang toolchains that ignore the
  # older -Xarch_device coverage split.
  target_compile_options(rccl-UnitTestsMicro PRIVATE --offload-host-only)
  target_link_options(rccl-UnitTestsMicro PRIVATE -no-hip-rt)

  target_compile_options(rccl-UnitTestsMicro PRIVATE
    -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicro PRIVATE
    -fprofile-instr-generate -fcoverage-mapping)

  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(rccl-UnitTestsMicro PRIVATE -fsanitize=address)
    target_link_options(rccl-UnitTestsMicro PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(rccl-UnitTestsMicro PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()

  # Deliberately NOT linking librccl.so: this binary is for tests that compile
  # their unit-under-test source file(s) directly. Add files to
  # RCCL_MICRO_TEST_SOURCES at the top of this file as new units come under test.
  rocm_install(TARGETS rccl-UnitTestsMicro COMPONENT tests)

  # Register with CTest exactly as rccl-UnitTests / rccl-UnitTestsFixtures do:
  # via RCCL's shared category mechanism (no bare add_test, matching the rest of
  # test/CMakeLists.txt). apply_test_category_labels is defined by
  # shared/ctest/TestCategories.cmake, which the parent test/CMakeLists.txt
  # include()s (globally) when building inside rocm-systems; INSTALL_TEST_FILE is
  # likewise set by the parent. Guard on COMMAND so a non-monorepo configure
  # (shared/ctest absent) still succeeds -- there, as with the other binaries,
  # the target is built but not category-registered.
  if(COMMAND apply_test_category_labels)
    set(_micro_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
    if(EXISTS "${_micro_categories_yaml}")
      message(STATUS "Applying test categories for rccl-UnitTestsMicro")
      apply_test_category_labels(
        rccl-UnitTestsMicro
        "${_micro_categories_yaml}"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
      )
    else()
      message(WARNING "Skipping test categories for rccl-UnitTestsMicro: missing ${_micro_categories_yaml}")
    endif()
  endif()

  # ---------------------------------------------------------------------------
  # rccl-UnitTestsMicroInit(+-uncached): host-only microtests for src/init.cc
  # (AICOMRCCL-1685). Dedicated executables (NOT shared with rccl-UnitTestsMicro)
  # so init.cc's symbols and std::once_flag state stay isolated. They also compile
  # the real host-only oracle TUs (argcheck/archinfo/utils) from the hipified
  # snapshot; --gc-sections drops their unreferenced deep deps. Two
  # HIP_*_UNCACHED_MEMORY variants exercise both arms of the cache-setting paths;
  # tests self-select via #ifdef and coverage is merged across the two binaries.
  # ---------------------------------------------------------------------------
  set(TEST_MICRO_INIT_SOURCE_FILES
    init-test.cc
    fakes/init_fakes.cc                  # init-only seams (ResetInitFakes)
    fakes/bootstrap_stubs.cc             # fail-loud stub floor, grouped by subsystem
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc                  # core/lifecycle stubs + data symbols
    fakes/nccl_fakes.cc                  # reusable nccl* stubs
    fakes/hip_fakes.cc                   # HIP runtime seams
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/env_fakes.cc                   # src/misc/param.cc + getenv interposition
    fakes/rccl_wrap_fakes.cc             # src/rccl_wrap.cc
    fakes/recorder_fakes.cc              # src/recorder.cc
    fakes/strongstream_stubs.cc          # src/misc/strongstream.cc
    fakes/tuning_fakes.cc                # src/graph/tuning.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/argcheck.cc  # real PtrCheck/CommCheck
    ${PROJECT_BINARY_DIR}/hipify/src/misc/archinfo.cc  # real IsArchMatch/GcnArchNameFormat
    ${PROJECT_BINARY_DIR}/hipify/src/misc/utils.cc     # real busId/hash helpers
    ../common/main_altrsmi.cpp
    ../common/ProcessIsolatedTestRunner.cpp
  )
  # The oracle TUs are produced by the hipify step (add_dependencies below), so
  # they don't exist at configure time -- mark GENERATED so a fresh configure
  # doesn't error on missing sources.
  set_source_files_properties(
    ${PROJECT_BINARY_DIR}/hipify/src/misc/argcheck.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/archinfo.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/utils.cc
    PROPERTIES GENERATED TRUE)

  # Two binaries, not one: init.cc gates HIP_HOST_UNCACHED_MEMORY/
  # HIP_UNCACHED_MEMORY at the *preprocessor* (#if defined / #ifndef, e.g. the
  # gfx950 corruption guard and the uncached-vs-finegrained malloc). Each arm's
  # code is absent from the object unless its macro is set, so a single compile
  # of init.cc can only ever reach one side -- covering both requires compiling
  # it twice with different -D. The base and -uncached binaries together exercise
  # both arms; coverage merges across their two .profraw files.
  foreach(_variant "" "-uncached")
    set(_init_tgt "rccl-UnitTestsMicroInit${_variant}")
    add_executable(${_init_tgt} ${TEST_MICRO_INIT_SOURCE_FILES})
    target_include_directories(${_init_tgt} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${RCCL_COMMON_INCLUDE_DIRS})
    # src/init.cc is the first of the duplicate basenames (vs transport/net_ib/
    # init.cc), so hipify keeps its name -- no _tmp suffix.
    target_compile_definitions(${_init_tgt} PRIVATE ${RCCL_COMMON_COMPILE_DEFS}
      INIT_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/init.cc")
    if(ENABLE_ROCSHMEM_GIN)
      target_compile_definitions(${_init_tgt} PRIVATE NCCL_GIN_ANVIL_SDMA_ENABLE=0)
    endif()
    if(_variant STREQUAL "-uncached")
      target_compile_definitions(${_init_tgt} PRIVATE HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY)
    endif()
    # HIP compile requirements without linking the runtime (see rccl-UnitTestsMicro).
    foreach(_hip_tgt hip::host hip::device)
      if(TARGET ${_hip_tgt})
        target_include_directories(${_init_tgt} PRIVATE
          $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${_init_tgt} PRIVATE
          $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
        target_compile_options(${_init_tgt} PRIVATE
          $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
      endif()
    endforeach()
    target_link_libraries(${_init_tgt} PRIVATE ${MICRO_TEST_LINK_LIBS})
    add_dependencies(${_init_tgt} hipify_all copy_nccl_device_headers)
    # Host-only (no amdgcn pass, so llvm coverage never reaches the device link) +
    # -ffunction-sections/--gc-sections so the oracle TUs' unreferenced deep deps
    # are dropped rather than pulling in the world.
    target_compile_options(${_init_tgt} PRIVATE
      --offload-host-only -ffunction-sections
      -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(${_init_tgt} PRIVATE
      -no-hip-rt
      -fprofile-instr-generate -fcoverage-mapping
      -Wl,--gc-sections)
    if(BUILD_ADDRESS_SANITIZER)
      target_compile_options(${_init_tgt} PRIVATE -fsanitize=address)
      target_link_options(${_init_tgt} PRIVATE -fsanitize=address -shared-libasan)
      if(DEFINED ASAN_RUNTIME_DIR)
        target_link_options(${_init_tgt} PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
      endif()
    endif()
    rocm_install(TARGETS ${_init_tgt} COMPONENT tests)
    if(COMMAND apply_test_category_labels)
      set(_init_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_init.yaml")
      if(EXISTS "${_init_categories_yaml}")
        message(STATUS "Applying test categories for ${_init_tgt}")
        apply_test_category_labels(
          ${_init_tgt}
          "${_init_categories_yaml}"
          "${PROJECT_BINARY_DIR}"
          "${INSTALL_TEST_FILE}"
        )
      else()
        message(WARNING "Skipping test categories for ${_init_tgt}: missing ${_init_categories_yaml}")
      endif()
    endif()
  endforeach()

  # ---------------------------------------------------------------------------
  # rccl-UnitTestsMicroEnqueue -- host-only microtests for src/enqueue.cc.
  #
  # Its own binary, not shared with the init/p2p ones: #include-ing a second
  # production .cc into an existing TU would collide on file-scope state
  # (enqueue.cc has 41 statics and 16 NCCL_PARAM/RCCL_PARAM globals).
  #
  # Two symbols must be omitted from the shared nccl_stubs.cc for this target:
  # enqueue.cc defines ncclInitKernelsForDevice and ncclParamGraphStreamOrdering
  # itself. Each is omitted by its OWN RCCL_STUBS_OMIT_<symbol> macro rather than
  # one target-wide switch, so the exclusion names exactly what it drops. (These
  # are still target-wide definitions -- a source added here later sees both --
  # but each names a single symbol.) An omit macro is only ever for a symbol the
  # UNIT UNDER TEST defines; a target that needs a real value where the floor
  # aborts gets a seam in the owning TU's fakes file instead.
  # ---------------------------------------------------------------------------
  # LINKS ONLY BECAUSE OF --gc-sections. Relinking without it leaves 78 undefined
  # symbols: 42 control globals that only init_fakes.cc defines (referenced from
  # the four *_stubs.cc below, never from enqueue-test.cc), and 36 ordinary
  # externals of enqueue.cc that nothing in fakes/ defines at all -- among them
  # hipFuncGetAttributes, netTransport, ncclGroupStartInternal and the
  # ncclProfiler* set. So a new test reaching one of those paths fails at LINK
  # time, not as an assertion: e.g. ncclInitKernelsForDevice (enqueue.cc:110)
  # needs a hipFuncGetAttributes fake before it can be called at all.
  # Rehoming the 42 globals to their owning TUs' fakes files addresses under half
  # of this; the other 36 need fakes written.
  set(TEST_MICRO_ENQUEUE_SOURCE_FILES
    enqueue-test.cc
    fakes/enqueue_fakes.cc               # ResetEnqueueFakes chain only
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/ce_fakes.cc                    # src/ce_coll.cc
    fakes/collectives_fakes.cc           # src/collectives.cc name tables/helpers
    fakes/comm_fakes.cc                  # src/init.cc comm lifecycle
    fakes/dev_runtime_fakes.cc           # src/dev_runtime.cc
    fakes/env_fakes.cc                   # src/misc/param.cc + getenv interposition
    fakes/os_fakes.cc                    # src/os/linux.cc aligned alloc
    fakes/proxy_fakes.cc                 # src/proxy.cc
    fakes/rccl_wrap_fakes.cc             # src/rccl_wrap.cc
    fakes/recorder_fakes.cc              # src/recorder.cc
    fakes/register_stubs.cc              # src/register/*.cc
    fakes/sched_stubs.cc                 # src/scheduler/*.cc + deep launch paths
    fakes/strongstream_stubs.cc          # src/misc/strongstream.cc
    fakes/sym_kernels_fakes.cc           # src/sym_kernels.cc
    fakes/tuning_fakes.cc                # src/graph/tuning.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/argcheck.cc  # real PtrCheck/CommCheck
    ${PROJECT_BINARY_DIR}/hipify/src/misc/archinfo.cc  # real IsArchMatch
    ${PROJECT_BINARY_DIR}/hipify/src/misc/utils.cc     # real busId/hash helpers
    ../common/main_altrsmi.cpp
  )

  # The oracle TUs are produced by the hipify step, so they do not exist at
  # configure time. The init block above marks them GENERATED, but source-file
  # properties are DIRECTORY-scoped -- relying on that is a hidden dependency on
  # target ordering within this file. Set it here too so this block stands alone.
  set_source_files_properties(
    ${PROJECT_BINARY_DIR}/hipify/src/misc/argcheck.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/archinfo.cc
    ${PROJECT_BINARY_DIR}/hipify/src/misc/utils.cc
    PROPERTIES GENERATED TRUE)

  add_executable(rccl-UnitTestsMicroEnqueue ${TEST_MICRO_ENQUEUE_SOURCE_FILES})
  target_include_directories(rccl-UnitTestsMicroEnqueue PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_COMMON_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/hipify/src/device          # device/common.h (neutered)
    ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device)
  # src/enqueue.cc is basename-unique in the tree, so hipify keeps its name --
  # no _tmp suffix (contrast transport/p2p.cc -> p2p_tmp.cc).
  target_compile_definitions(rccl-UnitTestsMicroEnqueue PRIVATE
    ${RCCL_COMMON_COMPILE_DEFS}
    # Per-symbol omissions from the shared nccl_stubs.cc; enqueue.cc itself
    # supplies both, so neither needs a replacement definition.
    RCCL_STUBS_OMIT_ncclInitKernelsForDevice
    RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
    ENQUEUE_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/enqueue.cc")
  # HIP compile requirements without linking the runtime (see rccl-UnitTestsMicro).
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-UnitTestsMicroEnqueue PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-UnitTestsMicroEnqueue PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
      target_compile_options(rccl-UnitTestsMicroEnqueue PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
    endif()
  endforeach()
  target_link_libraries(rccl-UnitTestsMicroEnqueue PRIVATE ${MICRO_TEST_LINK_LIBS})
  add_dependencies(rccl-UnitTestsMicroEnqueue hipify_all copy_nccl_device_headers)
  target_compile_options(rccl-UnitTestsMicroEnqueue PRIVATE
    --offload-host-only -ffunction-sections
    -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicroEnqueue PRIVATE
    -no-hip-rt
    -fprofile-instr-generate -fcoverage-mapping
    -Wl,--gc-sections)
  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(rccl-UnitTestsMicroEnqueue PRIVATE -fsanitize=address)
    target_link_options(rccl-UnitTestsMicroEnqueue PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(rccl-UnitTestsMicroEnqueue PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()
  rocm_install(TARGETS rccl-UnitTestsMicroEnqueue COMPONENT tests)
  if(COMMAND apply_test_category_labels)
    set(_enqueue_categories_yaml
        "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_enqueue.yaml")
    if(EXISTS "${_enqueue_categories_yaml}")
      message(STATUS "Applying test categories for rccl-UnitTestsMicroEnqueue")
      apply_test_category_labels(
        rccl-UnitTestsMicroEnqueue
        "${_enqueue_categories_yaml}"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
      )
    else()
      message(WARNING "Skipping test categories for rccl-UnitTestsMicroEnqueue: missing ${_enqueue_categories_yaml}")
    endif()
  endif()

  # ---------------------------------------------------------------------------
  # rccl-UnitTestsMicroDevRuntime: host-only microtests for src/dev_runtime.cc
  # (AICOMRCCL-2180). A dedicated executable rather than more sources on
  # rccl-UnitTestsMicro, for the same reason rccl-UnitTestsMicroInit is separate:
  # dev_runtime.cc drives the HIP VMM API directly, so fakes/dev_runtime_micro_fakes.cc
  # defines ~25 hip* entry points (hipMemCreate, hipMemMap, hipStream*, ...) that
  # fakes/hip_fakes.cc also defines. Linking both into one binary is a hard
  # multiple-definition error, and the two sets are not interchangeable: this
  # suite needs a real mmap-backed flat VA and per-call controllable failure,
  # where hip_fakes returns fixed canned values. Merging them would mean
  # rewriting shared fakes used by four other suites to no one's benefit.
  set(TEST_MICRO_DEVRUNTIME_SOURCE_FILES
    dev-runtime-test.cc
    fakes/dev_runtime_micro_fakes.cc
    ../common/main_altrsmi.cpp
    ../common/ProcessIsolatedTestRunner.cpp
  )

  add_executable(rccl-UnitTestsMicroDevRuntime ${TEST_MICRO_DEVRUNTIME_SOURCE_FILES})

  target_include_directories(rccl-UnitTestsMicroDevRuntime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device  # gin/gin_host.h
    ${PROJECT_BINARY_DIR}/hipify/src/device
    ${RCCL_COMMON_INCLUDE_DIRS})

  target_compile_definitions(rccl-UnitTestsMicroDevRuntime PRIVATE
    ${RCCL_COMMON_COMPILE_DEFS}
    DEV_RUNTIME_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/dev_runtime.cc")

  # HIP compile requirements without linking the runtime (see rccl-UnitTestsMicro).
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-UnitTestsMicroDevRuntime PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-UnitTestsMicroDevRuntime PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
      target_compile_options(rccl-UnitTestsMicroDevRuntime PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
    endif()
  endforeach()

  target_link_libraries(rccl-UnitTestsMicroDevRuntime PRIVATE ${MICRO_TEST_LINK_LIBS})
  add_dependencies(rccl-UnitTestsMicroDevRuntime
    hipify_all copy_nccl_device_headers
    net_ib_header_symlinks rma_header_symlinks)

  target_compile_options(rccl-UnitTestsMicroDevRuntime PRIVATE
    --offload-host-only
    -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicroDevRuntime PRIVATE
    -no-hip-rt
    -fprofile-instr-generate -fcoverage-mapping)

  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(rccl-UnitTestsMicroDevRuntime PRIVATE -fsanitize=address)
    target_link_options(rccl-UnitTestsMicroDevRuntime PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(rccl-UnitTestsMicroDevRuntime PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()

  rocm_install(TARGETS rccl-UnitTestsMicroDevRuntime COMPONENT tests)

  if(COMMAND apply_test_category_labels)
    set(_devr_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_devruntime.yaml")
    if(EXISTS "${_devr_categories_yaml}")
      message(STATUS "Applying test categories for rccl-UnitTestsMicroDevRuntime")
      apply_test_category_labels(
        rccl-UnitTestsMicroDevRuntime
        "${_devr_categories_yaml}"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
      )
    else()
      message(WARNING "Skipping test categories for rccl-UnitTestsMicroDevRuntime: missing ${_devr_categories_yaml}")
    endif()
  endif()
  return()
endif()

# --- hipcc is mandatory ---
# Auto-detect hipcc via ROCM_PATH, or pass -DCMAKE_CXX_COMPILER=hipcc.
if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
  if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
  endif()
  find_program(_hipcc hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin NO_CMAKE_FIND_ROOT_PATH)
  if(NOT _hipcc)
    find_program(_hipcc hipcc)
  endif()
  if(_hipcc)
    set(CMAKE_CXX_COMPILER "${_hipcc}")
  else()
    message(FATAL_ERROR
      "hipcc not found. Either:\n"
      "  cmake -B build -DCMAKE_CXX_COMPILER=/path/to/hipcc\n"
      "  cmake -B build -DROCM_PATH=/opt/rocm-6.x")
  endif()
endif()

# hipcc needs --gcc-install-dir to find C++ standard headers.
find_program(_system_gxx g++ REQUIRED)
execute_process(
  COMMAND ${_system_gxx} -print-file-name=libstdc++.a
  OUTPUT_VARIABLE _gcc_libstdcpp
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(_gcc_install_dir "${_gcc_libstdcpp}" DIRECTORY)
set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_gcc_install_dir} --offload-host-only")

project(rccl-HostUnitTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wno-tautological-constant-out-of-range-compare)

find_package(Threads REQUIRED)
include(FetchContent)

# --- Resolve GTest: system -> RCCL-vendored -> FetchContent fallback ---
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  set(_rccl_gtest_root "${CMAKE_CURRENT_SOURCE_DIR}/../../build/gtest")
  if(EXISTS "${_rccl_gtest_root}/lib/libgtest.a" OR EXISTS "${_rccl_gtest_root}/lib64/libgtest.a")
    message(STATUS "Using RCCL-vendored GTest: ${_rccl_gtest_root}")
    find_package(GTest REQUIRED PATHS "${_rccl_gtest_root}" NO_DEFAULT_PATH)
  else()
    message(STATUS "Downloading GTest via FetchContent")
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
endif()

# Both find_package(GTest) code paths above (system and RCCL-vendored) export
# only the modern GTest::gtest target; GTest::GTest is a legacy alias CMake's
# own FindGTest module stopped providing. Add it once here instead of after
# every branch so target_link_libraries(... GTest::GTest) below works
# regardless of which resolution path was taken.
if(TARGET GTest::gtest AND NOT TARGET GTest::GTest)
  # Promote before aliasing: GTest's package config creates GTest::gtest as a
  # non-GLOBAL imported target, and ALIAS of one of those needs CMake 3.18,
  # above this project's declared 3.16 minimum. Promotion works from 3.11 and
  # only from the directory that created the target, which is this file.
  set_target_properties(GTest::gtest PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(GTest::GTest ALIAS GTest::gtest)
endif()

# --- Resolve fmt: system -> FetchContent fallback ---
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "Downloading fmt via FetchContent")
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()

# --- Resolve RCCL build directory ---
set(RCCL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH
    "RCCL build directory (auto-detects release/debug subdirs)")

if(NOT EXISTS "${RCCL_BUILD_DIR}/hipify")
  foreach(_subdir release debug Release Debug RelWithDebInfo)
    if(EXISTS "${RCCL_BUILD_DIR}/${_subdir}/hipify")
      set(RCCL_BUILD_DIR "${RCCL_BUILD_DIR}/${_subdir}")
      message(STATUS "Detected build subdir layout: ${RCCL_BUILD_DIR}")
      break()
    endif()
  endforeach()
endif()

set(RCCL_HIPIFY_DIR "${RCCL_BUILD_DIR}/hipify")
set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

if(NOT EXISTS "${RCCL_HIPIFY_DIR}/src/graph/search.cc")
  message(FATAL_ERROR "Hipified sources not found at ${RCCL_HIPIFY_DIR}. Run the RCCL build first.")
endif()
message(STATUS "hipcc: ${CMAKE_CXX_COMPILER}")
message(STATUS "Hipified sources: ${RCCL_HIPIFY_DIR}")

# --- Detect whether the system amdsmi.h exposes the UALoE fabric struct ---
# Compile-only (STATIC_LIBRARY) avoids link failures from amdsmi symbols.
include(CheckCXXSourceCompiles)
unset(AMDSMI_FABRIC_API CACHE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles("
  #include <amd_smi/amdsmi.h>
  int main() {
    amdsmi_fabric_info_t info{};
    (void)info;
    return 0;
  }
" AMDSMI_FABRIC_API)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
if(AMDSMI_FABRIC_API)
  message(STATUS "amdsmi fabric API found - enabling AMDSMI_FABRIC_DIRECT")
else()
  message(STATUS "amdsmi fabric API not found (AMDSMI_FABRIC_DIRECT disabled)")
endif()

# --- Detect HIP fabric API support (mirrors top-level CMakeLists.txt) ---
# mem_manager.h only defines its own hipMemFabricHandle_st fallback when
# HIP_FABRIC_API is undefined; on HIP runtimes that already provide the type
# (e.g. ROCm 7.14+), omitting this define causes a redefinition conflict
# with hip_runtime_api.h.
# Compile-only (STATIC_LIBRARY), like the amdsmi probe above: with the default
# EXECUTABLE target type these checks link, which drags in the HIP runtime at
# configure time even though the binary itself is linked with -no-hip-rt.
include(CheckSymbolExists)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_symbol_exists("hipMemImportFromShareableHandle" "hip/hip_runtime_api.h" HIP_FABRIC_API_FUNC)
check_cxx_source_compiles("
  #include <hip/hip_runtime_api.h>
  int main() {
    hipMemFabricHandle_t handle;
    (void)handle;
    return 0;
  }
" HIP_FABRIC_HANDLE_TYPE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
  set(HIP_FABRIC_API ON)
  message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
else()
  set(HIP_FABRIC_API OFF)
endif()

# --- Common include paths for hipified RCCL headers ---
set(RCCL_INCLUDE_DIRS
  ${RCCL_HIPIFY_DIR}/src/include
  ${RCCL_HIPIFY_DIR}/src/include/plugin
  ${RCCL_HIPIFY_DIR}/src/include/plugin/tuner
  ${RCCL_BUILD_DIR}/include
  ${RCCL_HIPIFY_DIR}/src
  ${RCCL_HIPIFY_DIR}/src/graph
  ${RCCL_TEST_DIR}
  ${RCCL_TEST_DIR}/common
)

# --- accl-profiler plugin directory ---
set(ACCL_PROFILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins/profiler/accl")
set(RCCL_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# --- Wrapper library for production .cc files that need link stubs ---
add_library(rccl-source-wrappers STATIC
  mem_manager_src_wrapper.cc
  mem_manager_wrapper_test.cc
  bootstrap_wrapper.cc
  init_stubs.cpp
  wrapper_link_stubs.cpp
  accl_profiler_wrapper.cc
)
target_include_directories(rccl-source-wrappers PRIVATE
  ${ACCL_PROFILER_DIR}
  ${RCCL_INCLUDE_DIRS}
)
# accl_profiler_wrapper.cc needs the non-hipified profiler headers;
# scope the extra include to just that file to avoid os.h/rocmwrap.h
# conflicts with the hipified sources used by other wrappers.
set_source_files_properties(accl_profiler_wrapper.cc PROPERTIES
  COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
)
target_compile_definitions(rccl-source-wrappers PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
  $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
)
target_link_libraries(rccl-source-wrappers PRIVATE
  GTest::GTest
  fmt::fmt-header-only
)

# --- Main test executable ---
add_executable(rccl-HostUnitTests
  main.cpp
  ${RCCL_TEST_DIR}/BitOpsTests.cpp
  ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
  ${RCCL_TEST_DIR}/VersionInfoTests.cpp
  ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
  ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
  RomeTopoConsensusLogTests.cpp
  ${RCCL_TEST_DIR}/MiscTests.cpp
  ${RCCL_TEST_DIR}/VersionGateTests.cpp
  ${RCCL_TEST_DIR}/TimeoutTests.cpp
  ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
  ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
  ${RCCL_TEST_DIR}/AltRsmiTests.cpp
  ${RCCL_TEST_DIR}/AcclProfilerTests.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
  ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
  ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
  ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
)

target_include_directories(rccl-HostUnitTests PRIVATE
  ${ACCL_PROFILER_DIR}
  ${RCCL_INCLUDE_DIRS}
)
set_source_files_properties(${RCCL_TEST_DIR}/AcclProfilerTests.cpp PROPERTIES
  COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
)

target_compile_definitions(rccl-HostUnitTests PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  RCCL_EXPOSE_STATIC
  ARSMI_TEST_BUILD
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
  $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
)

# llvm source-based coverage (host-only build: no -Xarch_device split needed
# because --offload-host-only never runs the amdgcn device pass; see the
# CMAKE_CXX_FLAGS_INIT above). On by default so the binary always produces a
# profile; disable with -DHOST_UT_COVERAGE=OFF.
#
# RCCL_TEST_CODE_COVERAGE is what makes process-isolated tests flush their
# counters in the forked child (test/common/ProcessIsolatedTestRunner.cpp).
# Without it those tests run but contribute nothing to the profile, so it is
# tied to the same switch rather than exposed separately.
# rccl-source-wrappers must be instrumented too: mem_manager.cc reaches the
# binary through that static library, so instrumenting only the executable
# silently drops it from the report -- the profile still looks healthy, it just
# has no record of that file at all.
option(HOST_UT_COVERAGE "Build rccl-HostUnitTests with llvm source-based coverage" ON)
if(HOST_UT_COVERAGE)
  target_compile_options(rccl-HostUnitTests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-HostUnitTests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_compile_definitions(rccl-HostUnitTests PRIVATE RCCL_TEST_CODE_COVERAGE=1)
  target_compile_options(rccl-source-wrappers PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

target_link_options(rccl-HostUnitTests PRIVATE -no-hip-rt -no-pie)

target_link_libraries(rccl-HostUnitTests PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
  # RCCL_TEST_CODE_COVERAGE compiles a dlsym() call in
  # ProcessIsolatedTestRunner.cpp; the in-tree test build links dl for the same
  # runner. Empty on platforms where libdl is folded into libc (glibc >= 2.34),
  # required on older ones.
  ${CMAKE_DL_LIBS}
  -Wl,--whole-archive
  rccl-source-wrappers
  -Wl,--no-whole-archive
)

# --- Standalone micro-test binary ---------------------------------------
# rccl-UnitTestsMicro: same host-only model as rccl-HostUnitTests (hipcc
# --offload-host-only, -no-hip-rt), built from the hipified snapshot. It
# #includes each unit-under-test production .cc directly (via P2P_CC_PATH,
# RMA_PROXY_PROGRESS_CC_PATH, GROUP_CC_PATH, DEVCOMM_V22902_CC_PATH,
# DEVCOMM_V22907_CC_PATH) and
# satisfies every HIP/nccl symbol from fakes/ rather than librccl.so or the
# HIP runtime.
add_executable(rccl-UnitTestsMicro ${RCCL_MICRO_TEST_SOURCES})

target_compile_options(rccl-UnitTestsMicro PRIVATE
  -Wno-unused-parameter -Wno-unused-variable)

target_include_directories(rccl-UnitTestsMicro PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${RCCL_HIPIFY_DIR}/src/devcomm      # devcomm_v22902.h, included unqualified
  ${RCCL_INCLUDE_DIRS}
)

target_compile_definitions(rccl-UnitTestsMicro PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  P2P_CC_PATH="${RCCL_HIPIFY_DIR}/src/transport/p2p_tmp.cc"
  RMA_PROXY_PROGRESS_CC_PATH="${RCCL_HIPIFY_DIR}/src/rma/rma_proxy_progress.cc"
  GROUP_CC_PATH="${RCCL_HIPIFY_DIR}/src/group.cc"
  DEVCOMM_V22902_CC_PATH="${RCCL_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc"
  DEVCOMM_V22907_CC_PATH="${RCCL_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc"
  # Same fallback-vs-runtime conflicts the host target guards against: without
  # these, p2p-test.cc redefines hipMemFabricHandle_st and the amdsmi fabric
  # enumerators on ROCm 7.14+, and the standalone build fails.
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
  $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
)

# -no-pie for the same reason as rccl-HostUnitTests: the RCCL-vendored
# libgtest.a used by this standalone path is built without -fPIC, so a PIE
# link fails with "relocation R_X86_64_32 cannot be used against local symbol".
target_link_options(rccl-UnitTestsMicro PRIVATE -no-hip-rt -no-pie)
target_link_libraries(rccl-UnitTestsMicro PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
)

# llvm source-based coverage (host-only build: no -Xarch_device split needed
# because --offload-host-only never runs the amdgcn device pass).
# On by default; disable with -DMICRO_COVERAGE=OFF.
option(MICRO_COVERAGE "Build rccl-UnitTestsMicro with llvm source-based coverage" ON)
if(MICRO_COVERAGE)
  target_compile_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicro PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()

# --- Standalone init micro-test binaries --------------------------------
# rccl-UnitTestsMicroInit(+-uncached): host-only microtests for src/init.cc,
# same model as rccl-UnitTestsMicro. Compiles the real host-only oracle TUs
# (argcheck/archinfo/utils) from the hipified snapshot; --gc-sections drops
# their unreferenced deps. Two HIP_*_UNCACHED_MEMORY variants.
set(TEST_MICRO_INIT_SOURCE_FILES
  init-test.cc
  fakes/init_fakes.cc
  fakes/bootstrap_stubs.cc
  fakes/topo_stubs.cc
  fakes/transport_stubs.cc
  fakes/nccl_stubs.cc
  fakes/nccl_fakes.cc
  fakes/hip_fakes.cc
  fakes/env_fakes.cc
  fakes/rccl_wrap_fakes.cc
  fakes/recorder_fakes.cc
  fakes/strongstream_stubs.cc
  fakes/tuning_fakes.cc
  ${RCCL_HIPIFY_DIR}/src/misc/argcheck.cc
  ${RCCL_HIPIFY_DIR}/src/misc/archinfo.cc
  ${RCCL_HIPIFY_DIR}/src/misc/utils.cc
  ${RCCL_TEST_DIR}/common/main_altrsmi.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
)

# Two binaries, not one: init.cc gates HIP_HOST_UNCACHED_MEMORY/
# HIP_UNCACHED_MEMORY at the preprocessor, so each arm's code is absent from the
# object unless its macro is set. Covering both requires compiling init.cc twice
# with different -D (see the in-build block above for the full rationale).
foreach(_variant "" "-uncached")
  set(_init_tgt "rccl-UnitTestsMicroInit${_variant}")
  add_executable(${_init_tgt} ${TEST_MICRO_INIT_SOURCE_FILES})

  target_compile_options(${_init_tgt} PRIVATE
    -Wno-unused-parameter -Wno-unused-variable -ffunction-sections)

  target_include_directories(${_init_tgt} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_INCLUDE_DIRS}
  )

  target_compile_definitions(${_init_tgt} PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    # init.cc selects amdsmi_wrap.h (amd_smi/amdsmi.h, ships with ROCm) over the
    # legacy rocm_smi_wrap.h; the amd_smi_* wrapper API it declares is stubbed by
    # init_fakes.cc, so no SMI runtime is linked. The in-RCCL-build path inherits
    # this from RCCL_COMMON_COMPILE_DEFS instead.
    USE_AMDSMI
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
    INIT_CC_PATH="${RCCL_HIPIFY_DIR}/src/init.cc"
  )
  if(_variant STREQUAL "-uncached")
    target_compile_definitions(${_init_tgt} PRIVATE HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY)
  endif()

  target_link_options(${_init_tgt} PRIVATE -no-hip-rt -Wl,--gc-sections)
  target_link_libraries(${_init_tgt} PRIVATE
    GTest::GTest
    Threads::Threads
    fmt::fmt-header-only
  )

  if(MICRO_COVERAGE)
    target_compile_options(${_init_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(${_init_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  endif()
endforeach()


# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroEnqueue (standalone mode) -- see the in-build block for the
# rationale behind the separate binary and the RCCL_STUBS_OMIT_* macros.
# ---------------------------------------------------------------------------
# Same --gc-sections coupling as the in-build list; see the comment there.
set(TEST_MICRO_ENQUEUE_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/enqueue-test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/enqueue_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/bootstrap_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/topo_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/transport_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/nccl_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/nccl_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/hip_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/ce_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/collectives_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/comm_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/dev_runtime_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/env_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/os_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/proxy_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/rccl_wrap_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/recorder_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/register_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/sched_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/strongstream_stubs.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/sym_kernels_fakes.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fakes/tuning_fakes.cc
  ${RCCL_HIPIFY_DIR}/src/misc/argcheck.cc
  ${RCCL_HIPIFY_DIR}/src/misc/archinfo.cc
  ${RCCL_HIPIFY_DIR}/src/misc/utils.cc
  ${RCCL_TEST_DIR}/common/main_altrsmi.cpp
)

# Two binaries, for the same reason the init pair above exists: enqueue.cc gates
# rcclShmemDynamicSize (:72-88) at the PREPROCESSOR on RCCL_DEVICE_LINKER, so each
# arm's code is absent from the object unless its macro matches and a single
# compile can only ever reach one side.
#
# ENABLE_DEVICE_LINKER defaults ON, so the arm that SHIPS is the device-linker one.
# The in-RCCL-build path picks that up automatically (the rccl target's
# COMPILE_DEFINITIONS flow into RCCL_COMMON_COMPILE_DEFS), but this standalone
# project has no rccl target to read, and run_host_tests.sh -- the only in-repo
# workflow that runs these binaries -- uses this path. Without the second target
# the shipping arm would go untested by that workflow entirely.
foreach(_enq_variant "" "-devlinker")
  set(_enq_tgt "rccl-UnitTestsMicroEnqueue${_enq_variant}")
  add_executable(${_enq_tgt} ${TEST_MICRO_ENQUEUE_SOURCE_FILES})

  target_compile_options(${_enq_tgt} PRIVATE
    -Wno-unused-parameter -Wno-unused-variable -ffunction-sections)

  target_include_directories(${_enq_tgt} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RCCL_INCLUDE_DIRS}
    ${RCCL_HIPIFY_DIR}/src/device                # device/common.h (neutered)
    ${RCCL_HIPIFY_DIR}/src/include/nccl_device
  )

  target_compile_definitions(${_enq_tgt} PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    USE_AMDSMI
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
    RCCL_STUBS_OMIT_ncclInitKernelsForDevice
    RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
    ENQUEUE_CC_PATH="${RCCL_HIPIFY_DIR}/src/enqueue.cc"
  )
  if(_enq_variant STREQUAL "-devlinker")
    target_compile_definitions(${_enq_tgt} PRIVATE RCCL_DEVICE_LINKER)
  endif()

  target_link_options(${_enq_tgt} PRIVATE -no-hip-rt -Wl,--gc-sections)
  target_link_libraries(${_enq_tgt} PRIVATE
    GTest::GTest
    Threads::Threads
    fmt::fmt-header-only
  )

  if(MICRO_COVERAGE)
    target_compile_options(${_enq_tgt} PRIVATE
      -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(${_enq_tgt} PRIVATE
      -fprofile-instr-generate -fcoverage-mapping)
  endif()
endforeach()

# --- Standalone dev_runtime micro-test binary ---------------------------
# rccl-UnitTestsMicroDevRuntime: see the in-build block above for why this is
# its own executable rather than more sources on rccl-UnitTestsMicro.
add_executable(rccl-UnitTestsMicroDevRuntime
  dev-runtime-test.cc
  fakes/dev_runtime_micro_fakes.cc
  ${RCCL_TEST_DIR}/common/main_altrsmi.cpp
  ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
)

target_compile_options(rccl-UnitTestsMicroDevRuntime PRIVATE
  -Wno-unused-parameter -Wno-unused-variable)

target_include_directories(rccl-UnitTestsMicroDevRuntime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${RCCL_HIPIFY_DIR}/src/include/nccl_device   # gin/gin_host.h
  ${RCCL_HIPIFY_DIR}/src/device
  ${RCCL_INCLUDE_DIRS}
)

target_compile_definitions(rccl-UnitTestsMicroDevRuntime PRIVATE
  __HIP_PLATFORM_AMD__
  NCCL_OS_LINUX
  ROCM_VERSION=70000
  DEV_RUNTIME_CC_PATH="${RCCL_HIPIFY_DIR}/src/dev_runtime.cc"
  $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
  $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
)

# -no-pie for the same reason as the other standalone binaries (the
# RCCL-vendored libgtest.a is built without -fPIC).
target_link_options(rccl-UnitTestsMicroDevRuntime PRIVATE -no-hip-rt -no-pie)
target_link_libraries(rccl-UnitTestsMicroDevRuntime PRIVATE
  GTest::GTest
  Threads::Threads
  fmt::fmt-header-only
)

if(MICRO_COVERAGE)
  target_compile_options(rccl-UnitTestsMicroDevRuntime PRIVATE
    -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rccl-UnitTestsMicroDevRuntime PRIVATE
    -fprofile-instr-generate -fcoverage-mapping)
endif()

# NOTE: the standalone project intentionally does not register these binaries
# with CTest (matching test/host from #9320); run them directly, e.g.
#   ./build/rccl-UnitTestsMicro   or   ./build/rccl-HostUnitTests
#   ./build/rccl-UnitTestsMicroInit   ./build/rccl-UnitTestsMicroInit-uncached
#   ./build/rccl-UnitTestsMicroEnqueue  ./build/rccl-UnitTestsMicroEnqueue-devlinker
# The in-RCCL-build path registers the micro/init/enqueue binaries via CTest
# categories.
