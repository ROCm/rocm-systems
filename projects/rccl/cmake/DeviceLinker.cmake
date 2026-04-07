# cmake/DeviceLinker.cmake
#
# Assembly-extract device linker pipeline.
# Replaces the standard -fgpu-rdc compilation and linking of device code.
#
# All commands are expressed as add_custom_command so that the build system
# (ninja/make) can schedule them optimally alongside the rest of the build.
#
# Pipeline per specialized kernel (860 files, fully parallel):
#   compile .cpp -> .s  ->  extract .s -> extracted.s + .json  ->  assemble .o
#
# Dispatcher:
#   compile common.cu.cpp -> .s
#   aggregate resource .json files -> max_resources.json
#   patch dispatcher .s with max resources -> patched.s
#   assemble patched.s -> common_device.o
#
# Final:
#   lld -shared  ->  device.elf
#   clang-offload-bundler  ->  device.hipfb
#   host compile common.cu.cpp with -fcuda-include-gpubinary  ->  common.o (fat)
#   normal HIP compile onerank.cu.cpp  ->  onerank.o (fat)
#
# Required variables (set by including CMakeLists.txt):
#   HIPIFY_DIR, GEN_DIR, GPU_TARGETS, PROJECT_BINARY_DIR, PROJECT_SOURCE_DIR,
#   ROCM_PATH, Python3_EXECUTABLE

message(STATUS "Device Linker: assembly-extract pipeline enabled")

set(DEVICE_BUILD_DIR "${PROJECT_BINARY_DIR}/device_build")
set(ASM_EXTRACT_DIR  "${PROJECT_SOURCE_DIR}/tools/asm_extract")
set(SCRIPTS_DIR      "${PROJECT_SOURCE_DIR}/cmake/scripts")
set(SPECIALIZED_DIR  "${GEN_DIR}/specialized")

set(DL_CLANG "${ROCM_PATH}/bin/amdclang++")
set(DL_LLD "${ROCM_PATH}/llvm/bin/ld.lld")
set(DL_BUNDLER "${ROCM_PATH}/llvm/bin/clang-offload-bundler")

list(GET GPU_TARGETS 0 DL_GPU_RAW)
string(REGEX REPLACE ":.*" "" DL_GPU_TARGET "${DL_GPU_RAW}")
message(STATUS "Device Linker: GPU target = ${DL_GPU_TARGET}")

# Create output directories at configure time
file(MAKE_DIRECTORY
  ${DEVICE_BUILD_DIR}/specialized_asm
  ${DEVICE_BUILD_DIR}/extracted_asm
  ${DEVICE_BUILD_DIR}/extracted_obj
  ${DEVICE_BUILD_DIR}/resources
)

# ---------------------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------------------
set(DL_COMPILE_DEFS "")
if(COLLTRACE)
  list(APPEND DL_COMPILE_DEFS -DENABLE_COLLTRACE)
endif()
if(FAULT_INJECTION)
  list(APPEND DL_COMPILE_DEFS -DENABLE_FAULT_INJECTION)
endif()
if(LL128_ENABLED)
  list(APPEND DL_COMPILE_DEFS -DENABLE_LL128)
endif()
if(HIP_CONTIGUOUS_MEMORY)
  list(APPEND DL_COMPILE_DEFS -DHIP_CONTIGUOUS_MEMORY)
endif()
if(ENABLE_WARP_SPEED)
  list(APPEND DL_COMPILE_DEFS -DENABLE_WARP_SPEED)
endif()
if(PROFILE)
  list(APPEND DL_COMPILE_DEFS -DENABLE_PROFILING)
endif()
list(APPEND DL_COMPILE_DEFS
  -DFMT_HEADER_ONLY=1
  -DNCCL_MAJOR=${NCCL_MAJOR}
  -DNCCL_MINOR=${NCCL_MINOR}
  -DNCCL_PATCH=${NCCL_PATCH}
  -DNCCL_VERSION_CODE=${NCCL_VERSION_CODE}
  -DROCM_VERSION=${ROCM_VERSION}
  -D__HIP_PLATFORM_AMD__=1
)
if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  list(APPEND DL_COMPILE_DEFS -DHIP_UNCACHED_MEMORY)
endif()
if(HIP_HOST_UNCACHED_MEMORY)
  list(APPEND DL_COMPILE_DEFS -DHIP_HOST_UNCACHED_MEMORY)
endif()

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
set(DL_INCLUDE_DIRS
  -I${PROJECT_BINARY_DIR}/include
  -I${HIPIFY_DIR}/src
  -I${HIPIFY_DIR}/src/device
  -I${HIPIFY_DIR}/src/device/network/unpack
  -I${HIPIFY_DIR}/src/include
  -I${HIPIFY_DIR}/src/include/mlx5
  -I${HIPIFY_DIR}/src/include/nccl_device
  -I${HIPIFY_DIR}/src/include/ionic
  -I${HIPIFY_DIR}/src/include/plugin
  -I${GEN_DIR}
  -isystem${ROCM_PATH}/include
)

# fmt is needed by proxy_trace.h (included transitively from collectives.cc)
if(fmt_SOURCE_DIR)
  list(APPEND DL_INCLUDE_DIRS -isystem${fmt_SOURCE_DIR}/include)
else()
  get_target_property(_FMT_INC_DIRS fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
  if(_FMT_INC_DIRS)
    foreach(_dir ${_FMT_INC_DIRS})
      list(APPEND DL_INCLUDE_DIRS -isystem${_dir})
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# Optimization / common flags
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(DL_OPT_FLAGS -O0 -g)
else()
  set(DL_OPT_FLAGS -O3)
endif()

set(DL_DEVICE_COMPILE_FLAGS
  -x hip --cuda-device-only --offload-arch=${DL_GPU_TARGET}
  --no-gpu-bundle-output
  -gline-tables-only
  -std=c++17
  -w
  ${DL_OPT_FLAGS}
)

# ===========================================================================
# Per-specialized-kernel commands (compile -> extract -> assemble)
# ===========================================================================
set(SPECIALIZED_FILES_TXT "${GEN_DIR}/specialized_files.txt")
if(NOT EXISTS "${SPECIALIZED_FILES_TXT}")
  message(FATAL_ERROR "Device Linker: ${SPECIALIZED_FILES_TXT} not found. generate.py must run first.")
endif()

file(STRINGS "${SPECIALIZED_FILES_TXT}" SPECIALIZED_ENTRIES)
list(LENGTH SPECIALIZED_ENTRIES DL_KERNEL_COUNT)
message(STATUS "Device Linker: ${DL_KERNEL_COUNT} specialized kernels")

set(ALL_EXTRACTED_OBJS "")
set(ALL_RESOURCE_JSONS "")

foreach(ENTRY ${SPECIALIZED_ENTRIES})
  # Format: "filename funcname [guard]"
  string(REPLACE " " ";" ENTRY_LIST "${ENTRY}")
  list(GET ENTRY_LIST 0 CPP_FILE)
  string(REGEX REPLACE "\\.cpp$" "" BASE "${CPP_FILE}")

  set(SRC      "${SPECIALIZED_DIR}/${CPP_FILE}")
  set(ASM_OUT  "${DEVICE_BUILD_DIR}/specialized_asm/${BASE}.s")
  set(EXT_ASM  "${DEVICE_BUILD_DIR}/extracted_asm/${BASE}.s")
  set(RES_JSON "${DEVICE_BUILD_DIR}/resources/${BASE}.json")
  set(OBJ_OUT  "${DEVICE_BUILD_DIR}/extracted_obj/${BASE}.o")

  # Step 1: Compile specialized kernel to assembly
  add_custom_command(
    OUTPUT  ${ASM_OUT}
    COMMAND ${DL_CLANG}
      -DRCCL_DEVICE_LINKER
      ${DL_COMPILE_DEFS}
      ${DL_INCLUDE_DIRS}
      ${DL_DEVICE_COMPILE_FLAGS}
      -S
      -o ${ASM_OUT}
      ${SRC}
    DEPENDS ${SRC}
    COMMENT "DL compile: ${CPP_FILE}"
    VERBATIM
  )

  # Step 2: Extract device function + resource usage
  add_custom_command(
    OUTPUT  ${EXT_ASM} ${RES_JSON}
    COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/extract_device_function.py
      ${ASM_OUT} ${EXT_ASM} ${RES_JSON}
      > /dev/null
    DEPENDS ${ASM_OUT} ${ASM_EXTRACT_DIR}/extract_device_function.py
    COMMENT "DL extract: ${BASE}"
    VERBATIM
  )

  # Step 3: Assemble extracted function to relocatable object
  add_custom_command(
    OUTPUT  ${OBJ_OUT}
    COMMAND ${DL_CLANG}
      -x assembler -target amdgcn-amd-amdhsa -mcpu=${DL_GPU_TARGET}
      -c -o ${OBJ_OUT}
      ${EXT_ASM}
    DEPENDS ${EXT_ASM}
    COMMENT "DL assemble: ${BASE}"
    VERBATIM
  )

  list(APPEND ALL_EXTRACTED_OBJS ${OBJ_OUT})
  list(APPEND ALL_RESOURCE_JSONS ${RES_JSON})
endforeach()

# ===========================================================================
# Dispatcher: compile common.cu.cpp to device assembly
# ===========================================================================
set(COMMON_DEVICE_ASM "${DEVICE_BUILD_DIR}/common_device.s")

add_custom_command(
  OUTPUT  ${COMMON_DEVICE_ASM}
  COMMAND ${DL_CLANG}
    -DRCCL_DEVICE_LINKER
    -DUSE_INDIRECT_FUNCTION_CALL
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
    -x hip --cuda-device-only --offload-arch=${DL_GPU_TARGET}
    --no-gpu-bundle-output
    -g
    -std=c++17
    -w
    ${DL_OPT_FLAGS}
    -S
    -o ${COMMON_DEVICE_ASM}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/device/common.cu.cpp
  COMMENT "DL compile dispatcher: common.cu.cpp -> assembly (with -g)"
  VERBATIM
)

# ===========================================================================
# Aggregate resource usage across all specialized functions
# ===========================================================================
set(MAX_RESOURCES_JSON "${DEVICE_BUILD_DIR}/max_resources.json")

add_custom_command(
  OUTPUT  ${MAX_RESOURCES_JSON}
  COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/aggregate_resources.py
    ${DEVICE_BUILD_DIR}/resources
    ${MAX_RESOURCES_JSON}
  DEPENDS ${ALL_RESOURCE_JSONS} ${ASM_EXTRACT_DIR}/aggregate_resources.py
  COMMENT "DL aggregate: resource usage from ${DL_KERNEL_COUNT} functions"
  VERBATIM
)

# ===========================================================================
# Patch dispatcher assembly with aggregated resource values
# ===========================================================================
set(COMMON_DEVICE_PATCHED "${DEVICE_BUILD_DIR}/common_device_patched.s")

add_custom_command(
  OUTPUT  ${COMMON_DEVICE_PATCHED}
  COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/patch_dispatcher.py
    ${COMMON_DEVICE_ASM}
    ${COMMON_DEVICE_PATCHED}
    ${MAX_RESOURCES_JSON}
  DEPENDS ${COMMON_DEVICE_ASM} ${MAX_RESOURCES_JSON}
          ${ASM_EXTRACT_DIR}/patch_dispatcher.py
  COMMENT "DL patch dispatcher with max resources"
  VERBATIM
)

# ===========================================================================
# Assemble patched dispatcher
# ===========================================================================
set(COMMON_DEVICE_OBJ "${DEVICE_BUILD_DIR}/common_device.o")

add_custom_command(
  OUTPUT  ${COMMON_DEVICE_OBJ}
  COMMAND ${DL_CLANG}
    -x assembler -target amdgcn-amd-amdhsa -mcpu=${DL_GPU_TARGET}
    -c -o ${COMMON_DEVICE_OBJ}
    ${COMMON_DEVICE_PATCHED}
  DEPENDS ${COMMON_DEVICE_PATCHED}
  COMMENT "DL assemble: common_device.o"
  VERBATIM
)

# ===========================================================================
# Link all device objects into device.elf
# ===========================================================================
set(DEVICE_ELF "${DEVICE_BUILD_DIR}/device.elf")

add_custom_command(
  OUTPUT  ${DEVICE_ELF}
  COMMAND ${DL_LLD} -shared
    -o ${DEVICE_ELF}
    ${COMMON_DEVICE_OBJ}
    ${ALL_EXTRACTED_OBJS}
  DEPENDS ${COMMON_DEVICE_OBJ} ${ALL_EXTRACTED_OBJS}
  COMMENT "DL link: device.elf"
  VERBATIM
)

# ===========================================================================
# Bundle device.elf into a .hipfb fat binary
# ===========================================================================
set(DEVICE_HIPFB "${DEVICE_BUILD_DIR}/device.hipfb")

add_custom_command(
  OUTPUT  ${DEVICE_HIPFB}
  COMMAND ${DL_BUNDLER}
    --type=bc
    --targets=host-x86_64-unknown-linux-gnu-,hip-amdgcn-amd-amdhsa--${DL_GPU_TARGET}
    --input=/dev/null
    --input=${DEVICE_ELF}
    --output=${DEVICE_HIPFB}
  DEPENDS ${DEVICE_ELF}
  COMMENT "DL bundle: device.elf -> device.hipfb"
  VERBATIM
)

# ===========================================================================
# Host compile common.cu.cpp with embedded device binary
# ===========================================================================
set(COMMON_FAT_OBJ "${DEVICE_BUILD_DIR}/common.o")

add_custom_command(
  OUTPUT  ${COMMON_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-host-only --offload-arch=${DL_GPU_TARGET}
    -Xclang -fcuda-include-gpubinary -Xclang ${DEVICE_HIPFB}
    -DRCCL_DEVICE_LINKER
    -DUSE_INDIRECT_FUNCTION_CALL
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${COMMON_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  DEPENDS ${DEVICE_HIPFB} ${HIPIFY_DIR}/src/device/common.cu.cpp
  COMMENT "DL host compile: common.cu.cpp with embedded device binary"
  VERBATIM
)

# ===========================================================================
# Onerank: normal HIP compilation (host+device, no RDC)
# ===========================================================================
set(ONERANK_FAT_OBJ "${DEVICE_BUILD_DIR}/onerank.o")

add_custom_command(
  OUTPUT  ${ONERANK_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-arch=${DL_GPU_TARGET}
    -DRCCL_DEVICE_LINKER
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${ONERANK_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  COMMENT "DL compile: onerank.cu.cpp (normal fat object)"
  VERBATIM
)

# ===========================================================================
# collectives.cc: contains a __global__ kernel launch (hierarchicalAGShuffle)
# so it needs full HIP compilation, not --offload-host-only.
# ===========================================================================
set(COLLECTIVES_FAT_OBJ "${DEVICE_BUILD_DIR}/collectives.o")

add_custom_command(
  OUTPUT  ${COLLECTIVES_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-arch=${DL_GPU_TARGET}
    -DRCCL_DEVICE_LINKER
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${COLLECTIVES_FAT_OBJ}
    ${HIPIFY_DIR}/src/collectives.cc
  DEPENDS ${HIPIFY_DIR}/src/collectives.cc
  COMMENT "DL compile: collectives.cc (has __global__ kernel)"
  VERBATIM
)

# ===========================================================================
# Top-level target
# ===========================================================================
add_custom_target(device_linker_build ALL
  DEPENDS ${COMMON_FAT_OBJ} ${ONERANK_FAT_OBJ} ${COLLECTIVES_FAT_OBJ}
)
add_dependencies(device_linker_build hipify_all)

set(DEVICE_LINKER_OBJECTS
  ${COMMON_FAT_OBJ}
  ${ONERANK_FAT_OBJ}
  ${COLLECTIVES_FAT_OBJ}
)

# ===========================================================================
# Optional: emit LLVM IR for specialized kernels (ninja device_ir)
# ===========================================================================
set(DL_IR_DIR "${DEVICE_BUILD_DIR}/device_ir")
file(MAKE_DIRECTORY ${DL_IR_DIR})

set(ALL_IR_FILES "")
foreach(ENTRY ${SPECIALIZED_ENTRIES})
  string(REPLACE " " ";" ENTRY_LIST "${ENTRY}")
  list(GET ENTRY_LIST 0 CPP_FILE)
  string(REGEX REPLACE "\\.cpp$" "" BASE "${CPP_FILE}")

  set(SRC     "${SPECIALIZED_DIR}/${CPP_FILE}")
  set(IR_OUT  "${DL_IR_DIR}/${BASE}.ll")

  add_custom_command(
    OUTPUT  ${IR_OUT}
    COMMAND ${DL_CLANG}
      -DRCCL_DEVICE_LINKER
      ${DL_COMPILE_DEFS}
      ${DL_INCLUDE_DIRS}
      ${DL_DEVICE_COMPILE_FLAGS}
      -emit-llvm -S
      -o ${IR_OUT}
      ${SRC}
    DEPENDS ${SRC}
    COMMENT "DL IR: ${CPP_FILE}"
    VERBATIM
  )
  list(APPEND ALL_IR_FILES ${IR_OUT})
endforeach()

add_custom_target(device_ir DEPENDS ${ALL_IR_FILES})
add_dependencies(device_ir hipify_all)
