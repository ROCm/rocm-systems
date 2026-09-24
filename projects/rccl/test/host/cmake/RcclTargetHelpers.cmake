# Shared per-target helper functions for the host-only test build. These apply
# compile/link usage requirements and CTest registration that differ only by
# build mode, and are used by both the in-RCCL-build and standalone paths.

# Apply HIP's *compile* usage requirements (headers, device intrinsics/codegen,
# compile defs, compile options) to a target WITHOUT linking the HIP runtime.
# $<COMPILE_ONLY:hip::host> would express this in one line but requires
# CMake >= 3.27; pulling the interface properties directly keeps this valid at
# the project's declared minimum (3.16). Only the in-build micro targets use
# this (standalone builds with hipcc directly); rccl-source-wrappers applies a
# narrower variant inline because it deliberately omits the compile options.
function(rccl_apply_hip_compile_only _tgt)
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(${_tgt} PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(${_tgt} PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
      target_compile_options(${_tgt} PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_OPTIONS>)
    endif()
  endforeach()
endfunction()

# Mirrors the opt-out in test/CMakeLists.txt, whose loop does not reach these
# micro targets. Inheriting ENABLE_ROCSHMEM_GIN defaults NCCL_GIN_ANVIL_SDMA_ENABLE
# to 1, which pulls in rocSHMEM's sdma device headers. Only meaningful in-build
# where ENABLE_ROCSHMEM_GIN exists (standalone never defines it, so the guard is
# simply false there).
function(rccl_apply_rocshmem_gin_optout _tgt)
  if(ENABLE_ROCSHMEM_GIN)
    target_compile_definitions(${_tgt} PRIVATE NCCL_GIN_ANVIL_SDMA_ENABLE=0)
  endif()
endfunction()

# Mark the hipified "oracle" translation units (real argcheck/archinfo/utils, pulled
# in unhipified-source form) as GENERATED. They are produced by the hipify step
# (add_dependencies wires the ordering), so they do not exist at configure time --
# without this a fresh configure errors on the missing sources. Pass the mode's
# hipify base directory (${PROJECT_BINARY_DIR}/hipify in-build, ${RCCL_HIPIFY_DIR}
# standalone) and the oracle basenames under src/misc. Source-file properties are
# DIRECTORY-scoped, but we call this at each target so every block stands alone and
# does not rely on target ordering within this file.
function(rccl_mark_oracle_tus_generated _hipify_dir)
  set(_tus "")
  foreach(_basename ${ARGN})
    list(APPEND _tus "${_hipify_dir}/src/misc/${_basename}.cc")
  endforeach()
  set_source_files_properties(${_tus} PROPERTIES GENERATED TRUE)
endfunction()

# Register a micro target with CTest exactly as rccl-UnitTests / rccl-UnitTestsFixtures
# do: via RCCL's shared category mechanism (no bare add_test, matching the rest of
# test/CMakeLists.txt). apply_test_category_labels is defined by
# shared/ctest/TestCategories.cmake, which the parent test/CMakeLists.txt include()s
# (globally) when building inside rocm-systems; INSTALL_TEST_FILE is likewise set by
# the parent. Guard on COMMAND so a non-monorepo configure (shared/ctest absent) still
# succeeds -- there, as with the other binaries, the target is built but not
# category-registered. In-build only; the standalone block intentionally never
# registers with CTest.
function(rccl_register_test_categories _tgt _categories_yaml)
  if(COMMAND apply_test_category_labels)
    if(EXISTS "${_categories_yaml}")
      message(STATUS "Applying test categories for ${_tgt}")
      apply_test_category_labels(
        ${_tgt}
        "${_categories_yaml}"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
      )
    else()
      message(WARNING "Skipping test categories for ${_tgt}: missing ${_categories_yaml}")
    endif()
  endif()
endfunction()

# Apply the host-only compile/link flags shared by the in-build micro targets,
# plus the optional AddressSanitizer wiring.
function(rccl_apply_host_only_flags _tgt)
  target_compile_options(${_tgt} PRIVATE
    --offload-host-only -ffunction-sections
    -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(${_tgt} PRIVATE
    -no-hip-rt
    -fprofile-instr-generate -fcoverage-mapping
    -Wl,--gc-sections)
  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(${_tgt} PRIVATE -fsanitize=address)
    target_link_options(${_tgt} PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(${_tgt} PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()
endfunction()
