# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Blit kernels built offline for the ROCm backend, so the first blit doesn't compile them with
# comgr. The OpenCL C source is written by gen_blit_cl from the strings the runtime compiles
# (blitcl.cpp, rocblitcl.cpp); each target is compiled with the options the runtime uses for
# that device's default settings, and gen_blit_elf.py embeds the code objects in
# blit_kernels_elf.hpp. A device without a matching code object, or with settings that differ
# from the defaults, compiles the kernels at runtime as before.

set(_BLIT_LLVM_BIN "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/llvm/bin")
find_program(BLIT_CLANG_EXE NAMES amdclang clang HINTS ${_BLIT_LLVM_BIN} NO_DEFAULT_PATH)
find_package(Python3 COMPONENTS Interpreter)
if(NOT BLIT_CLANG_EXE OR NOT Python3_Interpreter_FOUND)
  message(WARNING "amdclang or Python 3 not found: blit kernels are compiled at runtime "
                  "(CLR_OFFLINE_BLIT_KERNELS=OFF)")
  set(CLR_OFFLINE_BLIT_KERNELS OFF CACHE BOOL "" FORCE)
  return()
endif()

# Writes the blit program source from the runtime's strings
add_executable(gen_blit_cl
  gen_blit_cl.cpp
  ${ROCCLR_SRC_DIR}/device/blitcl.cpp
  ${ROCCLR_SRC_DIR}/device/rocm/rocblitcl.cpp)
# The standard set on rocclr doesn't apply here, and MSVC defaults to C++14
target_compile_features(gen_blit_cl PRIVATE cxx_std_17)

# One code object per target, with the options roc::Device uses by default for it:
#   TAG  TARGET  CODE_OBJECT_VERSION  WAVE64  WGP_MODE  KERNARG_PRELOAD  GWS_INIT  IMAGES
# Generic targets need code object v6; gfx908 and gfx90a have no generic target. Preload
# follows Settings::setKernelArgImpl (gfx94x/gfx950, gfx125x); GWS init is supported before
# gfx94x and gfx11; WGP mode is on from gfx10, except on gfx125x. gfx125x has no image
# support, and the compiler can't build the image kernels for it.
set(BLIT_TARGETS
  "gfx9    gfx9-generic    6 1 0 0 1 1"
  "gfx908  gfx908          5 1 0 0 1 1"
  "gfx90a  gfx90a          5 1 0 0 1 1"
  "gfx9_4  gfx9-4-generic  6 1 0 1 0 1"
  "gfx10_1 gfx10-1-generic 6 0 1 0 1 1"
  "gfx10_3 gfx10-3-generic 6 0 1 0 1 1"
  "gfx11   gfx11-generic   6 0 1 0 0 1"
  "gfx11_7 gfx11-7-generic 6 0 1 0 0 1"
  "gfx12   gfx12-generic   6 0 1 0 0 1"
  "gfx12_5 gfx12-5-generic 6 0 0 1 0 0")

set(BLIT_ELFS "")
set(BLIT_GEN_ARGS "")
foreach(ENTRY ${BLIT_TARGETS})
  string(REGEX REPLACE " +" ";" FIELDS "${ENTRY}")
  list(GET FIELDS 0 TAG)
  list(GET FIELDS 1 TARGET)
  list(GET FIELDS 2 COV)
  list(GET FIELDS 3 WAVE64)
  list(GET FIELDS 4 WGP)
  list(GET FIELDS 5 PRELOAD)
  list(GET FIELDS 6 GWS)
  list(GET FIELDS 7 IMAGES)

  # Device::BlitProgram::create and Program::compileImpl: -O3, whole program, kernel arg info
  set(FLAGS -x cl -cl-std=CL1.2 -target amdgcn-amd-amdhsa "-mcpu=${TARGET}" -O3
            "-mcode-object-version=${COV}" -mllvm -amdgpu-prelink -cl-kernel-arg-info
            -mllvm -amdgpu-internalize-symbols -fno-rtlib-add-rpath "--rocm-path=${ROCM_PATH}")
  if(WAVE64)
    list(APPEND FLAGS -mwavefrontsize64)
  endif()
  if(NOT WGP)
    list(APPEND FLAGS -mcumode)
  endif()
  if(PRELOAD)
    list(APPEND FLAGS -Xclang -mllvm -Xclang -amdgpu-kernarg-preload-count=8)
  endif()
  set(SRC_ARGS "")
  if(GWS)
    list(APPEND SRC_ARGS gws)
  endif()
  if(IMAGES)
    list(APPEND SRC_ARGS images)
  endif()

  set(SRC "${CMAKE_CURRENT_BINARY_DIR}/blit_${TAG}.cl")
  add_custom_command(
    OUTPUT  "${SRC}"
    COMMAND gen_blit_cl "${SRC}" ${SRC_ARGS}
    DEPENDS gen_blit_cl
    COMMENT "Writing blit kernel source for ${TARGET}"
    VERBATIM)

  set(ELF "${CMAKE_CURRENT_BINARY_DIR}/blit_${TAG}.elf")
  add_custom_command(
    OUTPUT  "${ELF}"
    COMMAND "${BLIT_CLANG_EXE}" ${FLAGS} -o "${ELF}" "${SRC}"
    DEPENDS "${SRC}"
    COMMENT "Compiling blit kernels for ${TARGET}"
    VERBATIM)
  list(APPEND BLIT_ELFS "${ELF}")
  list(APPEND BLIT_GEN_ARGS "${TAG}" "${ELF}" ${WAVE64} ${WGP} ${PRELOAD} ${GWS} ${IMAGES})
endforeach()

set(BLIT_HPP "${CMAKE_CURRENT_BINARY_DIR}/blit_kernels_elf.hpp")
add_custom_command(
  OUTPUT  "${BLIT_HPP}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_blit_elf.py" "${BLIT_HPP}"
          ${BLIT_GEN_ARGS}
  DEPENDS ${BLIT_ELFS} "${CMAKE_CURRENT_SOURCE_DIR}/gen_blit_elf.py"
  COMMENT "Embedding blit kernel code objects"
  VERBATIM)
add_custom_target(blit_kernels_elf DEPENDS "${BLIT_HPP}")

add_dependencies(rocclr blit_kernels_elf)
target_include_directories(rocclr PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(rocclr PRIVATE ROCCLR_OFFLINE_BLIT_KERNELS=1)
