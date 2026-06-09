# >>> RCCL-DELTA: device build mechanism differs fundamentally between NCCL and RCCL.
#
# NCCL compiles device code as a CUDA `OBJECT` library (`nccl_device`) that the main
# library consumes via $<TARGET_OBJECTS:nccl_device>. RCCL instead:
#   * generates HIP device sources with the RCCL device generators into the hipify
#     staging tree (${GEN_DIR} / ${GEN_SYM_DIR}, defined by src/CMakeLists.txt before
#     this directory is added), and
#   * compiles them into librccl through the hipify pipeline + the custom device-linker
#     in src/CMakeLists.txt (common.cu / onerank.cu are hipified there, not listed here).
#
# The original NCCL content is preserved verbatim in the bracket comment below so future
# NCCL changes to this file remain easy to diff/merge; it is intentionally not executed.
#[==[ NCCL-ORIGINAL (disabled for RCCL)
# Run the scripts once during configuration to get the file lists
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    OUTPUT_VARIABLE files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${files}" files)
list(TRANSFORM files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    OUTPUT_VARIABLE symmetric_files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${symmetric_files}" symmetric_files)
list(TRANSFORM symmetric_files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)

# Create custom commands to generate source files with proper dependencies
add_custom_command(
    OUTPUT  ${files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating device source files"
)

add_custom_command(
    OUTPUT  ${symmetric_files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating symmetric device source files"
)

# Add library target
add_library(nccl_device OBJECT
            ${files}
            ${symmetric_files}
            ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
)

set_target_properties(nccl_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set include directories for the target
target_include_directories(nccl_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl_device nccl_header)
]==]

# RCCL device source generation (HIP). Runs the RCCL device + symmetric generators at
# configure time, writing into the shared hipify staging tree. The generated files are
# globbed and compiled (or routed through the device-linker) by src/CMakeLists.txt.
#
# These variables are provided by src/CMakeLists.txt before add_subdirectory(device):
#   RCCL_SOURCE_DIR, GEN_DIR, GEN_SYM_DIR
# and by the root CMakeLists: IFC_ENABLED, COLLTRACE, BUILD_LOCAL_GPU_TARGET_ONLY,
#   ENABLE_ROCSHMEM, ONLY_FUNCS, GENERATE_SYM_KERNELS.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "RCCL requires Python3 for generating host/device tables")
endif()

if(ONLY_FUNCS)
  message(WARNING "Using ONLY_FUNCS = ${ONLY_FUNCS}. Not meant for release builds.")
endif()

# Generate the collective functions / host + device tables.
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${GEN_DIR} ${IFC_ENABLED} ${COLLTRACE} OFF ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ENABLE_ROCSHMEM} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if(gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/generate.py failed")
endif()

if(GENERATE_SYM_KERNELS)
  # Generate the symmetric-memory kernels.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if(gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py failed")
  endif()
endif()
# <<< RCCL-DELTA

# >>> RCCL-DELTA: device/ headers staged for hipify. NCCL does not track headers in CMake
#     (and compiles device code as a CUDA OBJECT library), so RCCL lists the device headers
#     here and exports them for the hipify pipeline in src/CMakeLists.txt. The hand-written
#     device sources device/common.cu and device/onerank.cu remain in src/CMakeLists.txt
#     because they are routed through the RCCL device-linker filters there.
set(DEVICE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather.h
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_pivot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_gda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_gda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/op128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.h
    ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll128.h
    ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll.h
    ${CMAKE_CURRENT_SOURCE_DIR}/prims_simple.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rccl_metadata.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sendrecv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/hierarchical_ag_shuffle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack_defs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack.h
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather_gin.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_reduce.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/data_ops.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__funcs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/kernel.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/primitives.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter_gin.cuh
)
set(DEVICE_SOURCES ${DEVICE_SOURCES} PARENT_SCOPE)
# <<< RCCL-DELTA
