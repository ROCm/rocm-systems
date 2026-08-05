# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# rccl-UnitTestsMicro: standalone unit-test binary that we want to compile
# directly into the test binary. This lets us use preprocessor and link-time
# substitution for to provide test doubles for anything that would require
# a GPU or other special hardware.

if(NOT BUILD_TESTS)
  return()
endif()

set(TEST_MICRO_SOURCE_FILES
  p2p-test.cc
  fakes/nccl_fakes.cc                  # reusable stubs for nccl* symbols
  fakes/p2p_fakes.cc                   # p2p-specific stubs (arch/topo + alloc emulators)
  fakes/hip_fakes.cc                   # HIP runtime seams + link-satisfying stubs
  ../common/main_altrsmi.cpp           # minimal gtest main, no RCCL deps
  ../common/ProcessIsolatedTestRunner.cpp
)

add_executable(rccl-UnitTestsMicro ${TEST_MICRO_SOURCE_FILES})

target_include_directories(rccl-UnitTestsMicro PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
target_compile_definitions(rccl-UnitTestsMicro PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
target_link_libraries(rccl-UnitTestsMicro PRIVATE ${MICRO_TEST_LINK_LIBS})

# Pass the absolute path of hipified source files through as string
# literals so the #include in the test translation unit stays independent of
# the build directory layout.
target_compile_definitions(rccl-UnitTestsMicro PRIVATE
  P2P_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/transport/p2p_tmp.cc"
)

# Make sure the hipify step has run before trying to compile the test.
add_dependencies(rccl-UnitTestsMicro hipify_all copy_nccl_device_headers)

# Always build this binary with llvm source-based coverage.
# Restrict the instrumentation to the host compilation pass: applying it to the
# amdgcn device pass pulls in LLVM profiling runtime symbols
# (__llvm_profile_raw_version, __start___llvm_prf_cnts, ...) that don't exist on
# the device, which makes amdgcn-link fail.
target_compile_options(rccl-UnitTestsMicro PRIVATE
  -fprofile-instr-generate -fcoverage-mapping
  -Xarch_device -fno-profile-instr-generate
  -Xarch_device -fno-coverage-mapping)
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
# TEST_MICRO_SOURCE_FILES above as new units come under test.
rocm_install(TARGETS rccl-UnitTestsMicro COMPONENT tests)

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroInit: host-only microtests for src/init.cc
# (AICOMRCCL-1685). A DEDICATED executable (not shared with rccl-UnitTestsMicro)
# so init.cc's symbols don't collide with p2p's (allocTracker etc.), its
# std::once_flag state is isolated, and it can carry its own compile-definition
# variants. Deliberately does NOT link p2p-test.cc / p2p_fakes.cc.
#
# NOTE: real host-only oracle TUs (argcheck.cc / archinfo.cc / utils.cc) and the
# two HIP_*_UNCACHED_MEMORY compile variants are added during bring-up once the
# surviving-symbol inventory (steps 1-4) says which symbols must be satisfied.
# ---------------------------------------------------------------------------
set(TEST_MICRO_INIT_SOURCE_FILES
  init-test.cc
  fakes/init_fakes.cc                  # init-only seams (micro_getenv, ResetInitFakes)
  fakes/init_stubs.cc                  # fail-loud stubs (Tier-E) + data symbols
  fakes/nccl_fakes.cc                  # reusable nccl* stubs (g_loadParam, ...)
  fakes/hip_fakes.cc                   # HIP runtime seams
  # Real host-only oracle TUs (D3): compiled from the hipified snapshot so the
  # REAL PtrCheck/CommCheck (argcheck), IsArchMatch/GcnArchNameFormat (archinfo),
  # and busId/hash helpers (utils) are exercised, not reimplemented. --gc-sections
  # (below) drops every other function these TUs define so their deep deps don't
  # pull in the world.
  ${PROJECT_BINARY_DIR}/hipify/src/misc/argcheck.cc
  ${PROJECT_BINARY_DIR}/hipify/src/misc/archinfo.cc
  ${PROJECT_BINARY_DIR}/hipify/src/misc/utils.cc
  ../common/main_altrsmi.cpp           # minimal gtest main, no RCCL deps
  ../common/ProcessIsolatedTestRunner.cpp
)

add_executable(rccl-UnitTestsMicroInit ${TEST_MICRO_INIT_SOURCE_FILES})

target_include_directories(rccl-UnitTestsMicroInit PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
target_compile_definitions(rccl-UnitTestsMicroInit PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
target_link_libraries(rccl-UnitTestsMicroInit PRIVATE ${MICRO_TEST_LINK_LIBS})

# src/init.cc is the FIRST of the duplicate basenames (src/init.cc vs
# transport/net_ib/init.cc), so hipify keeps its name -- NO _tmp suffix.
target_compile_definitions(rccl-UnitTestsMicroInit PRIVATE
  INIT_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/init.cc"
)

add_dependencies(rccl-UnitTestsMicroInit hipify_all copy_nccl_device_headers)

# Coverage (host pass only; device pass disabled -- same rationale as above) +
# -ffunction-sections so the surviving-symbol probe link can --gc-sections.
target_compile_options(rccl-UnitTestsMicroInit PRIVATE
  -ffunction-sections
  -fprofile-instr-generate -fcoverage-mapping
  -Xarch_device -fno-profile-instr-generate
  -Xarch_device -fno-coverage-mapping)
target_link_options(rccl-UnitTestsMicroInit PRIVATE
  -fprofile-instr-generate -fcoverage-mapping
  # GC unreferenced functions (paired with -ffunction-sections) so the real
  # oracle TUs contribute only the handful of symbols init-test actually uses.
  -Wl,--gc-sections)

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl-UnitTestsMicroInit PRIVATE -fsanitize=address)
  target_link_options(rccl-UnitTestsMicroInit PRIVATE -fsanitize=address -shared-libasan)
  if(DEFINED ASAN_RUNTIME_DIR)
    target_link_options(rccl-UnitTestsMicroInit PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
  endif()
endif()

rocm_install(TARGETS rccl-UnitTestsMicroInit COMPONENT tests)
