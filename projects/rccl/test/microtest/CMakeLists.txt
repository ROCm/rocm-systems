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
  fakes/p2p_fakes.cc                   # stubs for everything p2p.cc references
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
