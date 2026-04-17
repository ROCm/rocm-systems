# Salmon transpiler lit + FileCheck harness. Optional: requires hipcc,
# clang-offload-bundler, FileCheck, not, and llvm-lit.

find_program(LIT_HIPCC_EXECUTABLE hipcc)
if(NOT LIT_HIPCC_EXECUTABLE)
  message(STATUS "transpiler lit tests skipped: hipcc not found")
  return()
endif()

set(LIT_CLANG_OFFLOAD_BUNDLER "")
foreach(
  _bundler_candidate
  "/opt/rocm/lib/llvm/bin/clang-offload-bundler"
  "/opt/rocm/llvm/bin/clang-offload-bundler"
)
  if(EXISTS "${_bundler_candidate}")
    set(LIT_CLANG_OFFLOAD_BUNDLER "${_bundler_candidate}")
    break()
  endif()
endforeach()
if(NOT LIT_CLANG_OFFLOAD_BUNDLER)
  find_program(
    LIT_CLANG_OFFLOAD_BUNDLER
    NAMES clang-offload-bundler
    PATHS /opt/rocm/lib/llvm/bin /opt/rocm/llvm/bin
  )
endif()
if(NOT LIT_CLANG_OFFLOAD_BUNDLER)
  message(STATUS "transpiler lit tests skipped: clang-offload-bundler not found")
  return()
endif()

set(LIT_FILECHECK_EXE "${LLVM_BINARY_DIR}/bin/FileCheck")
if(NOT EXISTS "${LIT_FILECHECK_EXE}")
  find_program(LIT_FILECHECK_EXE FileCheck HINTS "${LLVM_BINARY_DIR}/bin")
endif()
set(LIT_NOT_EXE "${LLVM_BINARY_DIR}/bin/not")
if(NOT EXISTS "${LIT_NOT_EXE}")
  find_program(LIT_NOT_EXE NAMES not HINTS "${LLVM_BINARY_DIR}/bin")
endif()
if(NOT LIT_FILECHECK_EXE OR NOT LIT_NOT_EXE)
  message(STATUS "transpiler lit tests skipped: FileCheck or not not found (LLVM tools)")
  return()
endif()

set(LIT_LLVM_LIT_EXE "${LLVM_BINARY_DIR}/bin/llvm-lit")
if(NOT EXISTS "${LIT_LLVM_LIT_EXE}")
  find_program(LIT_LLVM_LIT_EXE llvm-lit HINTS "${LLVM_BINARY_DIR}/bin")
endif()
if(NOT LIT_LLVM_LIT_EXE)
  message(STATUS "transpiler lit tests skipped: llvm-lit not found")
  return()
endif()

file(
  GLOB_RECURSE _LIT_HIP_RELS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hip"
)
list(SORT _LIT_HIP_RELS)

set(_LIT_ALL_COS "")
foreach(_hip_rel IN LISTS _LIT_HIP_RELS)
  set(_hip_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_hip_rel}")
  file(READ "${_hip_abs}" _hip_head LIMIT 200)
  set(_arch "gfx942")
  # CMake regex: avoid //[[:space:]]*... — leading // + POSIX classes can fail to match.
  if(_hip_head MATCHES "// *ARCH: *([A-Za-z0-9_]+)")
    set(_arch "${CMAKE_MATCH_1}")
  endif()

  get_filename_component(_stem "${_hip_rel}" NAME_WE)
  get_filename_component(_rel_dir "${_hip_rel}" DIRECTORY)
  if(_rel_dir STREQUAL "")
    set(_out_subdir ".")
  else()
    set(_out_subdir "${_rel_dir}")
  endif()

  set(_bundled_co
      "${CMAKE_CURRENT_BINARY_DIR}/${_out_subdir}/${_stem}_${_arch}.co")
  set(_unbundled_co
      "${CMAKE_CURRENT_BINARY_DIR}/${_out_subdir}/${_stem}_${_arch}_unbundled.co")

  add_custom_command(
    OUTPUT "${_unbundled_co}"
    COMMAND
      "${LIT_HIPCC_EXECUTABLE}" --genco "--offload-arch=${_arch}" -o
      "${_bundled_co}" "${_hip_abs}"
    COMMAND
      "${LIT_CLANG_OFFLOAD_BUNDLER}" --type=o "--input=${_bundled_co}"
      "--output=${_unbundled_co}"
      "--targets=hipv4-amdgcn-amd-amdhsa--${_arch}" --unbundle
    DEPENDS "${_hip_abs}"
    VERBATIM)

  list(APPEND _LIT_ALL_COS "${_unbundled_co}")

  string(TOUPPER "${_stem}" _stem_upper)
  set(LIT_FIXTURE_${_stem_upper}_CO "${_unbundled_co}")
endforeach()

add_custom_target(transpiler_lit_fixtures DEPENDS ${_LIT_ALL_COS})

set(LIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRANSPILER_BUILD_DIR "${CMAKE_BINARY_DIR}")
# Single-config Ninja: raise_cli lands next to this project in the build tree.
set(LIT_TOOL_RAISE_CLI "${CMAKE_BINARY_DIR}/raise_cli")
set(LIT_TOOL_FILECHECK "${LIT_FILECHECK_EXE}")
set(LIT_TOOL_NOT "${LIT_NOT_EXE}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

# Canonical LLVM convention for `check-*` targets: `-sv` = succinct
# (only per-test results on failure) + verbose-on-failure. Matches
# check-llvm / check-clang / check-mlir so CI logs stay readable.
add_custom_target(
  check-transpiler-lit
  COMMAND "${LIT_LLVM_LIT_EXE}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS raise_cli transpiler_lit_fixtures
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Running transpiler lit tests")

# CTest registration. The `transpiler-lit` test itself only invokes
# llvm-lit; it assumes the `.co` fixtures and the `raise_cli` binary
# already exist in the build tree. CTest does not rebuild targets on
# its own, so we pair the test with a FIXTURES_SETUP test that runs
# `cmake --build` for the required targets and a FIXTURES_REQUIRED
# back-reference, letting `ctest -R transpiler-lit` just work on a
# clean tree. The setup test is idempotent — if the fixtures already
# exist, the build is a no-op.
add_test(
  NAME transpiler-lit-build-fixtures
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
    --target transpiler_lit_fixtures raise_cli)
set_tests_properties(
  transpiler-lit-build-fixtures
  PROPERTIES FIXTURES_SETUP transpiler-lit-fixtures)

add_test(
  NAME transpiler-lit
  COMMAND "${LIT_LLVM_LIT_EXE}" -sv "${CMAKE_CURRENT_BINARY_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(
  transpiler-lit
  PROPERTIES FIXTURES_REQUIRED transpiler-lit-fixtures)
