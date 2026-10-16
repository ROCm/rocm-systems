# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

project(aie_tests)

enable_testing()

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa_runtime64 hsa-runtime64)

include(GoogleTest)

# test_error_callback: validates the per-queue error callback for AIE queues. Self-contained;
# it does not require any AIE kernel artifacts (no aiecc / NPU toolchain needed).
add_executable(test_error_callback test_error_callback.cc)
set_target_properties(test_error_callback
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(test_error_callback
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(test_error_callback
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
gtest_discover_tests(test_error_callback)

# test_memory: allocation, VMem and sharing tests for AIE agents. Self-contained;
# it does not require any AIE kernel artifacts (no aiecc / NPU toolchain needed).
add_executable(test_memory test_memory.cc)
set_target_properties(test_memory
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(test_memory
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(test_memory
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
gtest_discover_tests(test_memory)

# ---------------------------------------------------------------------------
# Everything below builds AIE kernel artifacts and the dispatch tests that need
# them. It is all conditional on the kernel toolchain being present, because
# test_error_callback and test_memory above need none of it and are expected to
# configure and run on any machine -- a missing NPU or IRON environment must not
# fail the configure and take those targets down with it.
#
# AIE_KERNEL_TOOLCHAIN is set to FALSE by the first missing piece; each probe
# below is skipped once it is false, so the reason reported is the first thing
# actually missing rather than the last one checked.
# ---------------------------------------------------------------------------
set(AIE_KERNEL_TOOLCHAIN TRUE)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(NOT Python3_FOUND)
  message(STATUS "Python3 interpreter not found; AIE kernel artifacts and the dispatch tests "
                 "will not be built.")
  set(AIE_KERNEL_TOOLCHAIN FALSE)
endif()

# ---------------------------------------------------------------------------
# NPU detection
# ---------------------------------------------------------------------------
if(AIE_KERNEL_TOOLCHAIN)
  # Only probe when the caller has not already said which target to build for; -DNPU_TARGET=...
  # is how you build kernels on a machine whose NPU is not the one you are targeting.
  find_program(XRT_SMI xrt-smi)
  if(XRT_SMI AND NOT NPU_TARGET)
    execute_process(
      COMMAND "${XRT_SMI}" examine
      OUTPUT_VARIABLE XRT_SMI_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(XRT_SMI_OUTPUT MATCHES "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]")
      set(NPU_TARGET "npu2")
    elseif(XRT_SMI_OUTPUT MATCHES "NPU Phoenix|RyzenAI-npu1")
      set(NPU_TARGET "npu")
    endif()
  endif()
  if(NOT NPU_TARGET)
    message(STATUS "Could not auto-detect an NPU device; AIE kernel artifacts and the dispatch "
                   "tests will not be built. Set NPU_TARGET to 'npu' or 'npu2' to force it.")
    set(AIE_KERNEL_TOOLCHAIN FALSE)
  else()
    message(STATUS "NPU target: ${NPU_TARGET}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Peano (llvm-aie) detection
#
# aiecc compiles the AIE cores with Peano, which ships as the llvm-aie wheel
# installed alongside mlir-aie in the IRON environment.
# ---------------------------------------------------------------------------
if(AIE_KERNEL_TOOLCHAIN)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, sysconfig; print(os.path.join(sysconfig.get_paths()['purelib'], 'llvm-aie'))"
    OUTPUT_VARIABLE PEANO_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT EXISTS "${PEANO_INSTALL_DIR}")
    message(STATUS "Could not find Peano (llvm-aie) at '${PEANO_INSTALL_DIR}'; AIE kernel artifacts "
                  "and the dispatch tests will not be built. Set up the IRON environment with "
                  "iron_env_setup.sh to build them.")
    set(AIE_KERNEL_TOOLCHAIN FALSE)
  else()
    message(STATUS "Peano install dir: ${PEANO_INSTALL_DIR}")
  endif()
endif()

# aie-hsaco packages the kernel artifacts into the hsacos the loader tests dispatch from. It ships
# with mlir-aie, next to aiecc.
if(AIE_KERNEL_TOOLCHAIN)
  find_program(AIE_HSACO aie-hsaco)
  if(NOT AIE_HSACO)
    message(STATUS "Could not find aie-hsaco; AIE kernel artifacts and the dispatch tests will not "
                   "be built. It ships with mlir-aie; set up the IRON environment to build them.")
    set(AIE_KERNEL_TOOLCHAIN FALSE)
  else()
    message(STATUS "aie-hsaco: ${AIE_HSACO}")
  endif()
endif()

if(NOT AIE_KERNEL_TOOLCHAIN)
  message(STATUS "AIE kernel toolchain incomplete: building test_error_callback and test_memory only.")
  return()
endif()

# ---------------------------------------------------------------------------
# Full-ELF toolchain detection
#
# Probes for the tools a full-ELF build needs and sets AIE_FULL_ELF_SUPPORTED.
# The build itself lives in add_aie_kernel below.
# ---------------------------------------------------------------------------

# xclbinutil and aiebu-asm ship alongside XRT; aiecc needs both on PATH to
# assemble the control code embedded in the full ELF.
find_program(XCLBINUTIL xclbinutil
  HINTS "/opt/xilinx/xrt/bin/unwrapped" "/opt/xilinx/xrt/bin"
)
find_program(AIEBU_ASM aiebu-asm
  HINTS "/opt/xilinx/xrt/bin" "/opt/xilinx/aiebu/bin"
)

# aiecc drives aiebu-asm with the 'aie2_config' target. Older aiebu releases
# only know that target as 'config', and aiebu-asm exits 0 on an unrecognised
# target -- so aiecc reports success and never writes an ELF. Detect that here
# rather than letting the build silently produce nothing.
# Full-ELF dispatch is aie2p only -- the design's npu_load_pdi has no aie2 equivalent -- so there
# is nothing to build for an npu1 target even with a complete toolchain.
set(AIE_FULL_ELF_SUPPORTED FALSE)
if(NPU_TARGET STREQUAL "npu2" AND XCLBINUTIL AND AIEBU_ASM)
  execute_process(
    COMMAND "${AIEBU_ASM}" --help
    OUTPUT_VARIABLE AIEBU_ASM_HELP
    ERROR_VARIABLE AIEBU_ASM_HELP
  )
  if(AIEBU_ASM_HELP MATCHES "aie2_config")
    set(AIE_FULL_ELF_SUPPORTED TRUE)
  endif()
endif()
if(NOT AIE_FULL_ELF_SUPPORTED)
  message(STATUS
    "Full-ELF kernel will not be built (needs an npu2 target plus xclbinutil and an aiebu-asm "
    "supporting the 'aie2_config' target); the full-ELF dispatch tests will skip themselves.")
endif()

# ---------------------------------------------------------------------------
# Kernel artifact build pipeline
#
# add_aie_kernel(<name> <source.py>) builds one IRON design two ways and sets, in
# the caller's scope:
#
#   <name>_INSTS_BIN         instruction sequence for the PDI dispatch path
#   <name>_DESIGN_PDI        PDI for that path
#   <name>_FULL_ELF          path the standalone ELF would occupy
#   <name>_FULL_ELF_TARGET   target building that ELF, or "" when unsupported
#
# <name>_FULL_ELF is always set, but the ELF is only built when
# AIE_FULL_ELF_SUPPORTED. That is deliberate: the tests check the file for
# existence and skip themselves when it is absent, so they need the path either
# way. Depend on <name>_FULL_ELF_TARGET rather than naming
# <name>_full_elf_artifacts directly -- it is empty when there is nothing to
# build, so a consumer cannot accidentally require a target that was never
# defined on a machine without the toolchain.
#
# The full-ELF build of a design differs from the PDI one: its runtime sequence
# starts with aiex.npu.load_pdi, because nothing configures the array for it out
# of band. Each build also gets its own directory keyed on <name>, because aiecc
# writes <input>.prj beside its input and two runs sharing a directory clobber
# each other's intermediates under 'make -j'.
# ---------------------------------------------------------------------------
# Extra arguments after <source.py> are forwarded to the design script for both builds.
function(add_aie_kernel name source)
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/kernel_${name}")
  set(mlir "${build_dir}/aie.mlir")
  set(insts "${build_dir}/insts.bin")
  set(pdi "${build_dir}/design.pdi")
  get_filename_component(source_name "${source}" NAME)

  # Step 1: Python -> MLIR
  add_custom_command(
    OUTPUT "${mlir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}"
    COMMAND "${Python3_EXECUTABLE}" "${source}" "${NPU_TARGET}" ${ARGN} > "${mlir}"
    DEPENDS "${source}"
    COMMENT "Generating aie.mlir from ${source_name} (target: ${NPU_TARGET})"
    VERBATIM
  )

  # Step 2: MLIR -> insts.bin + design.pdi (via aiecc)
  add_custom_command(
    OUTPUT "${insts}" "${pdi}"
    COMMAND aiecc
            --peano=${PEANO_INSTALL_DIR}
            --get-npu-insts
            --get-pdi
            --npu-insts-name=insts.bin
            --pdi-name=design.pdi
            "${mlir}"
    WORKING_DIRECTORY "${build_dir}"
    DEPENDS "${mlir}"
    COMMENT "Compiling AIE kernel ${source_name} with aiecc"
    VERBATIM
  )

  add_custom_target(${name}_kernel_artifacts DEPENDS "${insts}" "${pdi}")
  set(${name}_INSTS_BIN "${insts}" PARENT_SCOPE)
  set(${name}_DESIGN_PDI "${pdi}" PARENT_SCOPE)

  set(elf_build_dir "${CMAKE_CURRENT_BINARY_DIR}/kernel_full_elf_${name}")
  set(elf_mlir "${elf_build_dir}/aie.mlir")
  set(elf "${elf_build_dir}/aie.elf")
  set(${name}_FULL_ELF "${elf}" PARENT_SCOPE)
  if(NOT AIE_FULL_ELF_SUPPORTED)
    set(${name}_FULL_ELF_TARGET "" PARENT_SCOPE)
    return()
  endif()

  get_filename_component(XCLBINUTIL_DIR "${XCLBINUTIL}" DIRECTORY)
  get_filename_component(AIEBU_ASM_DIR "${AIEBU_ASM}" DIRECTORY)

  add_custom_command(
    OUTPUT "${elf_mlir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${elf_build_dir}"
    COMMAND "${Python3_EXECUTABLE}" "${source}" "${NPU_TARGET}" --full-elf ${ARGN} > "${elf_mlir}"
    DEPENDS "${source}"
    COMMENT "Generating full-ELF aie.mlir from ${source_name} (target: ${NPU_TARGET})"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${elf}"
    # aiebu-asm exits 0 even when it fails, and aiecc does not check that the
    # ELF appeared. Delete first so that a silent failure leaves nothing behind
    # and the build stops, instead of resurrecting the previous ELF.
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${elf}"
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${XCLBINUTIL_DIR}:${AIEBU_ASM_DIR}:$ENV{PATH}"
            aiecc
            --peano=${PEANO_INSTALL_DIR}
            --get-full-elf
            --full-elf-name=aie.elf
            "${elf_mlir}"
    # Errors out (non-zero) if aiecc produced no ELF.
    COMMAND "${CMAKE_COMMAND}" -E md5sum "${elf}"
    WORKING_DIRECTORY "${elf_build_dir}"
    DEPENDS "${elf_mlir}"
    COMMENT "Generating full ELF from AIE kernel ${source_name}"
    VERBATIM
  )

  add_custom_target(${name}_full_elf_artifacts DEPENDS "${elf}")
  set(${name}_FULL_ELF_TARGET ${name}_full_elf_artifacts PARENT_SCOPE)
endfunction()

# vector_scalar_add: two kernargs (separate input and output), adds 1.
add_aie_kernel(vsadd "${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_add.py")
# vector_scalar_mul: one kernarg (in-place buffer), multiplies by a constant.
# Its different result and different kernarg count are what let the interleaving
# tests tell the two designs apart.
# The scale lives here so the design and the tests' expected values cannot drift apart: the same
# value is compiled into the kernel and into the assertion that checks it.
set(VSMUL_SCALE 3)
add_aie_kernel(vsmul "${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_mul.py" "--scale=${VSMUL_SCALE}")

# ---------------------------------------------------------------------------
# Unified hsaco packaging
#
# Packages each design's artifacts into an AIE hsaco with aie-hsaco. test_dispatch.cc loads
# every kernel it dispatches from these through hsa_executable_load_agent_code_object.
#
# aie-hsaco's --arch takes "aie2"/"aie2p", while NPU_TARGET (detected above) is "npu"/"npu2",
# so map the latter onto the former.
# ---------------------------------------------------------------------------
if(NPU_TARGET STREQUAL "npu2")
  set(AIE_HSACO_ARCH "aie2p")
else()
  set(AIE_HSACO_ARCH "aie2")
endif()

set(VSADD_HSACO_KERNEL_NAME "vsadd")
set(vsadd_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd.hsaco")
add_custom_command(
  OUTPUT "${vsadd_HSACO}"
  COMMAND "${AIE_HSACO}"
          --hsaco "${vsadd_HSACO}"
          --arch "${AIE_HSACO_ARCH}"
          # vsadd's kernarg buffer is 2 pointers + 2 sizes (see
          # aie_vector_scalar_kernel::kernarg_bytes in test_dispatch.cc); the design uses a single
          # NPU column (NPU1Col1/NPU2Col1).
          --kernel "${VSADD_HSACO_KERNEL_NAME}:${vsadd_INSTS_BIN}:${vsadd_DESIGN_PDI}:32:1"
  # Depend on the target, not its output files: two custom commands racing aiecc in the same
  # build directory produce concurrent runs that clobber each other's intermediates.
  DEPENDS vsadd_kernel_artifacts
  COMMENT "Packaging vsadd PDI+insts artifacts into a unified AIE hsaco"
  VERBATIM
)
add_custom_target(vsadd_hsaco_artifact DEPENDS "${vsadd_HSACO}")

# The same packaging for the mul design, so the interleaving tests can dispatch two distinct
# kernels by handle. Its kernarg buffer is 1 pointer + 1 size (see
# aie_vector_scalar_mul_kernel::kernarg_bytes in test_dispatch.cc).
set(VSMUL_HSACO_KERNEL_NAME "vsmul")
set(vsmul_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsmul.hsaco")
add_custom_command(
  OUTPUT "${vsmul_HSACO}"
  COMMAND "${AIE_HSACO}"
          --hsaco "${vsmul_HSACO}"
          --arch "${AIE_HSACO_ARCH}"
          --kernel "${VSMUL_HSACO_KERNEL_NAME}:${vsmul_INSTS_BIN}:${vsmul_DESIGN_PDI}:16:1"
  # Depend on the target, not its output files: see the vsadd_HSACO rule above for why.
  DEPENDS vsmul_kernel_artifacts
  COMMENT "Packaging vsmul PDI+insts artifacts into a unified AIE hsaco"
  VERBATIM
)
add_custom_target(vsmul_hsaco_artifact DEPENDS "${vsmul_HSACO}")

# One hsaco with a section per arch, for aie_code_test's arch selection. aie2 is packed first so the
# aie2p section is not the first AIE section in the file. The payloads only have to parse, not run,
# so the artifacts of the detected device stand in for both arches.
set(multiarch_HSACO "${CMAKE_CURRENT_BINARY_DIR}/multiarch.hsaco")
add_custom_command(
  OUTPUT "${multiarch_HSACO}"
  # aie-hsaco adds to an existing file, so start from none.
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${multiarch_HSACO}"
  COMMAND "${AIE_HSACO}"
          --hsaco "${multiarch_HSACO}"
          --arch aie2
          --kernel "${VSMUL_HSACO_KERNEL_NAME}:${vsmul_INSTS_BIN}:${vsmul_DESIGN_PDI}:16:1"
  COMMAND "${AIE_HSACO}"
          --hsaco "${multiarch_HSACO}"
          --arch aie2p
          --kernel "${VSADD_HSACO_KERNEL_NAME}:${vsadd_INSTS_BIN}:${vsadd_DESIGN_PDI}:32:1"
  DEPENDS vsadd_kernel_artifacts vsmul_kernel_artifacts
  COMMENT "Packaging vsmul (aie2) and vsadd (aie2p) into a multi-arch AIE hsaco"
  VERBATIM
)
add_custom_target(multiarch_hsaco_artifact DEPENDS "${multiarch_HSACO}")

# Packages the vsadd full-ELF artifact into an AIE hsaco with kind=FullElf, for the full-ELF
# tests in test_dispatch.cc. The path is set unconditionally (mirroring vsadd_FULL_ELF above) so
# the tests' file-existence check has something to point at; the packaging command itself only
# runs when the toolchain can emit full ELFs.
set(vsadd_ELF_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd_elf.hsaco")
if(AIE_FULL_ELF_SUPPORTED)
  add_custom_command(
    OUTPUT "${vsadd_ELF_HSACO}"
    COMMAND "${AIE_HSACO}"
            --hsaco "${vsadd_ELF_HSACO}"
            --arch "${AIE_HSACO_ARCH}"
            --kernel "elf:${vsadd_FULL_ELF}:0:1"
    # Depend on the target, not its output files: see the vsadd_HSACO rule above for why.
    DEPENDS vsadd_full_elf_artifacts
    COMMENT "Packaging vsadd full-ELF artifact into a unified AIE hsaco"
    VERBATIM
  )
  add_custom_target(vsadd_elf_hsaco_artifact DEPENDS "${vsadd_ELF_HSACO}")
endif()

# The mul design's full ELF, packaged the same way, for the full-ELF interleaving test.
set(vsmul_ELF_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsmul_elf.hsaco")
if(AIE_FULL_ELF_SUPPORTED)
  add_custom_command(
    OUTPUT "${vsmul_ELF_HSACO}"
    COMMAND "${AIE_HSACO}"
            --hsaco "${vsmul_ELF_HSACO}"
            --arch "${AIE_HSACO_ARCH}"
            --kernel "elf:${vsmul_FULL_ELF}:0:1"
    # Depend on the target, not its output files: see the vsadd_HSACO rule above for why.
    DEPENDS vsmul_full_elf_artifacts
    COMMENT "Packaging vsmul full-ELF artifact into a unified AIE hsaco"
    VERBATIM
  )
  add_custom_target(vsmul_elf_hsaco_artifact DEPENDS "${vsmul_ELF_HSACO}")
endif()

# ---------------------------------------------------------------------------
# test_dispatch executable
# ---------------------------------------------------------------------------
add_executable(test_dispatch test_dispatch.cc)
set_target_properties(test_dispatch
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(test_dispatch
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
# core/inc is not installed alongside the runtime; needed for amd_aie_section.h, a header-only
# dependency used to locate and corrupt kernel-table fields in the negative tests.
target_include_directories(test_dispatch PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime)
target_compile_definitions(test_dispatch
  PRIVATE
    MUL_SCALE=${VSMUL_SCALE}
    DEFAULT_HSACO_PATH=${vsadd_HSACO}
    MULTIARCH_HSACO_PATH=${multiarch_HSACO}
    DEFAULT_HSACO_KERNEL_NAME="${VSADD_HSACO_KERNEL_NAME}"
    DEFAULT_ELF_HSACO_PATH=${vsadd_ELF_HSACO}
    # The kernel is addressed as "<device>:<sequence>", which aie-hsaco's elf: form derives from
    # the nested ELF. Neither design names its aie.device or its runtime sequence, so aiecc
    # defaults them to "main" and "sequence" for both.
    DEFAULT_ELF_HSACO_KERNEL_NAME="main:sequence"
    MUL_HSACO_PATH=${vsmul_HSACO}
    MUL_HSACO_KERNEL_NAME="${VSMUL_HSACO_KERNEL_NAME}"
    MUL_ELF_HSACO_PATH=${vsmul_ELF_HSACO}
)
target_link_libraries(test_dispatch
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
add_dependencies(test_dispatch vsadd_kernel_artifacts vsmul_kernel_artifacts vsadd_hsaco_artifact
                               vsmul_hsaco_artifact multiarch_hsaco_artifact)
# Both are empty when the toolchain cannot build full ELFs. The guard is needed because
# add_dependencies() rejects a call with no dependencies, but it tests the variables themselves
# rather than AIE_FULL_ELF_SUPPORTED, so adding a kernel here cannot name a target that the
# function never defined.
if(vsadd_FULL_ELF_TARGET OR vsmul_FULL_ELF_TARGET)
  add_dependencies(test_dispatch ${vsadd_FULL_ELF_TARGET} ${vsmul_FULL_ELF_TARGET})
endif()
if(AIE_FULL_ELF_SUPPORTED)
  add_dependencies(test_dispatch vsadd_elf_hsaco_artifact vsmul_elf_hsaco_artifact)
endif()

gtest_discover_tests(test_dispatch)

# ---------------------------------------------------------------------------
# elf_parse_test: exercises the in-tree full-ELF reader (core/inc/amd_aie_elf.h)
# against the vsadd full-ELF artifact. The parser source is compiled directly
# into the test because core/inc is not installed alongside the runtime.
# ---------------------------------------------------------------------------
add_executable(elf_parse_test
  elf_parse_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/core/runtime/amd_aie_elf.cpp)
set_target_properties(elf_parse_test
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(elf_parse_test
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_include_directories(elf_parse_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime)
target_link_libraries(elf_parse_test PRIVATE GTest::gtest_main hsa-runtime64::hsa-runtime64)
if(vsadd_FULL_ELF_TARGET)
  add_dependencies(elf_parse_test ${vsadd_FULL_ELF_TARGET})
endif()
add_test(NAME elf_parse_test COMMAND elf_parse_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------------------------------------------------------
# aie_code_test: exercises the in-tree unified-hsaco reader (core/inc/amd_aie_code.hpp)
# against vsadd.hsaco, plus garbage/truncated inputs. Host-side only: needs neither hardware
# nor an AIE agent, unlike test_dispatch's HsacoKernelObjectIsPublished/AieKindIsValidated. The
# parser and its amd::elf::Image dependency are compiled directly into the test for the same
# reason as elf_parse_test's amd_aie_elf.cpp: core/inc is not installed, and amd::elf::Image is
# built with hidden visibility inside libhsa-runtime64.so, so it cannot be linked against from
# outside it.
add_executable(aie_code_test
  aie_code_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/core/runtime/amd_aie_code.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/libamdhsacode/amd_elf_image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/libamdhsacode/amd_hsa_code_util.cpp)
set_target_properties(aie_code_test
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_include_directories(aie_code_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hsa-runtime/libamdhsacode)
target_compile_definitions(aie_code_test PRIVATE AIE_HSACO_ARCH="${AIE_HSACO_ARCH}")
target_link_libraries(aie_code_test PRIVATE GTest::gtest_main elf)
add_dependencies(aie_code_test vsadd_hsaco_artifact multiarch_hsaco_artifact)
add_test(NAME aie_code_test COMMAND aie_code_test WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
