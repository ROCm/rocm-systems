# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# rccl-UnitTestsMch: standalone unit-test binary for files under
# src/transport/ that we want to compile directly into the test binary
# (rather than reach via librccl.so). This lets us link-time-substitute
# the HIP driver / NCCL proxy seams for hermetic, GPU-less coverage.
#
# Pattern modelled on rccl-UnitTestsAltRsmi in ../CMakeLists.txt.

if(NOT BUILD_TESTS)
  return()
endif()

set(TEST_MCH_SOURCE_FILES
  p2p-test.cc
  fakes/p2p_fakes.cc                   # stubs for everything p2p.cc references
  ../common/main_altrsmi.cpp           # minimal gtest main, no RCCL deps
  ../common/ProcessIsolatedTestRunner.cpp
)

add_executable(rccl-UnitTestsMch ${TEST_MCH_SOURCE_FILES})

target_include_directories(rccl-UnitTestsMch PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
target_compile_definitions(rccl-UnitTestsMch PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
target_link_libraries(rccl-UnitTestsMch PRIVATE ${RCCL_COMMON_LINK_LIBS})

# Point p2p-test.cc at the hipified copy of p2p.cc that the main RCCL build
# generates (cudaXxx -> hipXxx, CUdeviceptr -> hipDeviceptr_t, etc.). We
# pass the absolute path through as a string literal so the #include in
# the test TU stays independent of the build directory layout.
target_compile_definitions(rccl-UnitTestsMch PRIVATE
  P2P_CC_PATH="${PROJECT_BINARY_DIR}/hipify/src/transport/p2p.cc"
)

# Make sure the hipify step has produced the file before we try to compile
# the test. We depend on the `rccl` target (which owns the hipify outputs)
# WITHOUT linking against it -- the test deliberately compiles p2p.cc
# directly and provides its own stubs for the rest.
add_dependencies(rccl-UnitTestsMch rccl)

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl-UnitTestsMch PRIVATE -fsanitize=address)
  target_link_options(rccl-UnitTestsMch PRIVATE -fsanitize=address -shared-libasan)
  if(DEFINED ASAN_RUNTIME_DIR)
    target_link_options(rccl-UnitTestsMch PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
  endif()
endif()

# Deliberately NOT linking librccl.so: this binary is for tests that compile
# their unit-under-test source file(s) directly. Add files to
# TEST_MCH_SOURCE_FILES above as new units come under test.

rocm_install(TARGETS rccl-UnitTestsMch COMPONENT tests)
