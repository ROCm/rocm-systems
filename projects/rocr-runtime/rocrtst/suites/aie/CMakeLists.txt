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

# error_callback_test: validates the per-queue error callback for AIE queues. Self-contained;
# it does not require any AIE kernel artifacts (no aiecc / NPU toolchain needed).
add_executable(error_callback_test error_callback_test.cc)
set_target_properties(error_callback_test
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(error_callback_test
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(error_callback_test
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
gtest_discover_tests(error_callback_test)

# ---------------------------------------------------------------------------
# Everything below builds AIE kernel artifacts and the dispatch tests that need
# them. It is all conditional on the kernel toolchain being present, because
# error_callback_test above needs none of it and is expected to configure and
# run on any machine -- a missing NPU or IRON environment must not fail the
# configure and take that target down with it.
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

if(NOT AIE_KERNEL_TOOLCHAIN)
  message(STATUS "AIE kernel toolchain incomplete: building error_callback_test only.")
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
# dispatch test executable
# ---------------------------------------------------------------------------
add_executable(dispatch dispatch.cc)
set_target_properties(dispatch
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(dispatch
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_compile_definitions(dispatch
  PRIVATE
    DEFAULT_PDI_PATH=${vsadd_DESIGN_PDI}
    DEFAULT_INSTS_PATH=${vsadd_INSTS_BIN}
    # The kernel is addressed as "<device>:<sequence>", the names aiecc writes
    # into aie.mlir.prj/full_elf_config.json. Neither design names its
    # aie.device or its runtime sequence, so aiecc defaults them to "main" and
    # "sequence" for both.
    DEFAULT_ELF_PATH=${vsadd_FULL_ELF}
    DEFAULT_ELF_KERNEL_NAME="main:sequence"
    MUL_PDI_PATH=${vsmul_DESIGN_PDI}
    MUL_INSTS_PATH=${vsmul_INSTS_BIN}
    MUL_ELF_PATH=${vsmul_FULL_ELF}
    MUL_SCALE=${VSMUL_SCALE}
)
target_link_libraries(dispatch
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
add_dependencies(dispatch vsadd_kernel_artifacts vsmul_kernel_artifacts)
# Both are empty when the toolchain cannot build full ELFs. The guard is needed because
# add_dependencies() rejects a call with no dependencies, but it tests the variables themselves
# rather than AIE_FULL_ELF_SUPPORTED, so adding a kernel here cannot name a target that the
# function never defined.
if(vsadd_FULL_ELF_TARGET OR vsmul_FULL_ELF_TARGET)
  add_dependencies(dispatch ${vsadd_FULL_ELF_TARGET} ${vsmul_FULL_ELF_TARGET})
endif()

gtest_discover_tests(dispatch)
