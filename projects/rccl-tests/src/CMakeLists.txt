# ########################################################################
# Copyright 2022-2026 Advanced Micro Devices, Inc.
# ########################################################################

function(add_rccl_test TEST)
    set(TEST_SOURCE "${TEST}.cu")
    set_property(SOURCE ${TEST_SOURCE} PROPERTY LANGUAGE CXX)

    # Check that file exists
    if (NOT EXISTS ${SOURCE_DIR}/${TEST_SOURCE})
        message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${SOURCE_DIR}/${TEST_SOURCE}")
    endif()

    # Establish hipified copy of the source file
    set(HIP_FILE "${HIPIFY_DIR}/${TEST_SOURCE}")
    get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

    # Convert .cu files to .cpp so that they get processed properly
    string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})

    # Create a custom command to create hipified source code
    add_custom_command(
        OUTPUT ${HIP_FILE}
        COMMAND mkdir -p ${HIP_FILE_DIR} && $ ${hipify-perl_executable} -quiet-warnings ${SOURCE_DIR}/${TEST_SOURCE} -o ${HIP_FILE}
        MAIN_DEPENDENCY ${TEST_SOURCE}
        COMMENT "Hipifying ${TEST_SOURCE} -> ${HIP_FILE}"
    )

    set(TEST_TARGET "${TEST}_perf")
    add_executable(${TEST_TARGET} ${HIP_FILE})
    target_link_libraries(
        ${TEST_TARGET}
        PRIVATE
            rccl_common
    )
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
            # LINKER_LANGUAGE CXX
    )
    add_relative_test(${TEST} ${TEST_TARGET})
    rocm_install(TARGETS ${TEST_TARGET})
    # TODO: copy/install DLLs on Windows
    set_target_properties(
        ${TEST_TARGET} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib"
    )
endfunction()

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

# Collect list of common source files
#==================================================================================================
set(COMMON_FILES
  git_version.h
  collector.h
  collector.cu
  common.h
  common.cu
  nccl1_compat.h
  rccl_compat.h
  rccl_float8.h
  rccl_vector_types.h
  multimem_ops.h
  timer.h
  timer.cc
  util.h
  util.cu
  ../verifiable/verifiable.h
  ../verifiable/verifiable.cu
)

# Hipify common files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
set(HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

## Loop over each common file to hipify
foreach(COMMON_FILE ${COMMON_FILES})
  # Check that file exists
  if (NOT EXISTS ${SOURCE_DIR}/${COMMON_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${SOURCE_DIR}/${COMMON_FILE}")
  endif()

  # Establish hipified copy of the common file
  get_filename_component(HIP_FILE_NAME ${HIPIFY_DIR}/${COMMON_FILE} NAME)
  set(HIP_FILE "${HIPIFY_DIR}/${HIP_FILE_NAME}")

  # Convert .cu files to .cpp so that they get processed properly
  string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  list(APPEND HIP_COMMON_SOURCES ${HIP_FILE})

  # Create a custom command to create hipified source code
  add_custom_command(
    OUTPUT ${HIP_FILE}
    COMMAND mkdir -p ${HIPIFY_DIR} && $ ${hipify-perl_executable} -quiet-warnings ${SOURCE_DIR}/${COMMON_FILE} -o ${HIP_FILE}
    MAIN_DEPENDENCY ${COMMON_FILE}
    COMMENT "Hipifying ${COMMON_FILE} -> ${HIP_FILE}"
  )
endforeach()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")
list(APPEND HIP_COMMON_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

#Create a custom target that updates git_version.cpp and executes whenever rccl is built
add_custom_target(git_version_check
  COMMENT "Updating git_version.cpp if necessary"
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  VERBATIM
)

# Compile common object library
#==================================================================================================
add_custom_target(hipify DEPENDS ${HIP_COMMON_SOURCES})
add_library(rccl_common OBJECT ${HIP_COMMON_SOURCES})
add_dependencies(rccl_common hipify git_version_check)
target_link_libraries(rccl_common roc::rccl hip::device Threads::Threads dl hsakmt)
if(USE_MPI)
  target_link_libraries(rccl_common MPI::MPI_CXX)
endif()

# Compile tests
#==================================================================================================
add_rccl_test(all_gather)
add_rccl_test(all_reduce)
add_rccl_test(alltoall)
if (ENABLE_ROCSHMEM AND ROCSHMEM_LIBRARY)
    target_compile_definitions(alltoall_perf PRIVATE ENABLE_ROCSHMEM ENABLE_ROCSHMEM_GIN)
    target_compile_options(alltoall_perf PRIVATE -fgpu-rdc $<$<CONFIG:Debug>:-ggdb -O0>)
    find_library(_DRM drm HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
    find_library(_DRM_AMDGPU drm_amdgpu HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
    target_link_libraries(alltoall_perf PRIVATE
        -Wl,--whole-archive
        ${ROCSHMEM_LIBRARY}
        -Wl,--no-whole-archive
        hsakmt ibverbs hsa-runtime64 numa ${_DRM} ${_DRM_AMDGPU})
    set(_alltoall_device_defines "")
    foreach(_gpu_raw ${GPU_TARGETS})
        string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
        list(APPEND _alltoall_device_defines "-Xarch_${_gpu}=-DENABLE_ROCSHMEM" "-Xarch_${_gpu}=-DENABLE_ROCSHMEM_GIN")
    endforeach()
    # --export-dynamic: librccl.so (GIN-only build) resolves rocshmem/anvil host symbols from this exe
    target_link_options(alltoall_perf PRIVATE -fgpu-rdc --hip-link ${_rocshmem_offload_arch_flags} -Wl,--export-dynamic ${_alltoall_device_defines})
endif()
add_rccl_test(alltoallv)
add_rccl_test(broadcast)
add_rccl_test(gather)
add_rccl_test(hypercube)
add_rccl_test(reduce_scatter)
add_rccl_test(reduce)
add_rccl_test(scatter)
add_rccl_test(sendrecv)
add_rccl_test(all_reduce_bias)
