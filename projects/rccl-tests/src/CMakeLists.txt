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
    rccl_tests_hipify("${HIP_FILE}" "${SOURCE_DIR}/${TEST_SOURCE}")

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
    if(WIN32)
      add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:${TEST_TARGET}> $<TARGET_FILE_DIR:${TEST_TARGET}>
        COMMAND_EXPAND_LISTS VERBATIM)
      install(FILES $<TARGET_RUNTIME_DLLS:${TEST_TARGET}> DESTINATION bin)
      set_tests_properties(${TEST} PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:hip::amdhip64>")
    endif()
    set_target_properties(
        ${TEST_TARGET} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib"
    )
endfunction()

function(add_relative_test test_name test_target)
    add_test(NAME "${test_name}" COMMAND ${test_target} -b 8 -e 1M -f 2 -g 1 -d float -o sum -n 5 -w 1 -c 1)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 120 LABELS "gpu" RESOURCE_LOCK rccl_tests_gpu)
endfunction()

# Collect list of common source files
#==================================================================================================
set(COMMON_FILES
  git_version.h
  collector.h
  collector.cu
  alltoall_hier_policy.h
  common.h
  common.cu
  gin_sdma_allgather_policy.h
  gin_sdma_allgather_reqs.h
  gin_sdma_broadcast_policy.h
  gin_sdma_devtime.h
  gin_sdma_devtime_host.h
  gin_resource_sharing.h
  nccl1_compat.h
  rccl_compat.h
  library.h
  rccl_float8.h
  rccl_vector_types.h
  multimem_ops.h
  timer.h
  timer.cc
  util.h
  util.cu
)

# Hipify common files (copy of source generated into hipify directory)
#==================================================================================================
set(HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

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
  rccl_tests_hipify("${HIP_FILE}" "${SOURCE_DIR}/${COMMON_FILE}")
endforeach()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp" "const char *rcclTestsGitHash = \"Unknown\";\n")
endif()
list(APPEND HIP_COMMON_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

#Create a custom target that updates git_version.cpp and executes whenever rccl is built
add_custom_target(git_version_check
  COMMENT "Updating git_version.cpp if necessary"
  COMMAND ${CMAKE_COMMAND} "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  VERBATIM
)

# Compile common object library
#==================================================================================================
add_custom_target(hipify DEPENDS ${HIP_COMMON_SOURCES})
add_library(rccl_common OBJECT ${HIP_COMMON_SOURCES})
add_dependencies(rccl_common hipify git_version_check)
target_link_libraries(rccl_common roc::rccl hip::device Threads::Threads ${CMAKE_DL_LIBS} nccl_test_os nccl_tests_options verifiable)
# dlsym(RTLD_DEFAULT) cannot see symbols of a statically linked RCCL.
get_target_property(RCCL_LIBRARY_TYPE roc::rccl TYPE)
if(RCCL_LIBRARY_TYPE STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(rccl_common PRIVATE RCCL_TESTS_STATIC_RCCL)
endif()
if(USE_MPI)
  target_link_libraries(rccl_common MPI::MPI_CXX)
endif()

# Compile tests
#==================================================================================================
add_rccl_test(all_gather)
add_rccl_test(all_reduce)
add_rccl_test(alltoall)
add_rccl_test(alltoallv)
add_rccl_test(broadcast)
add_rccl_test(gather)
add_rccl_test(hypercube)
add_rccl_test(reduce_scatter)
add_rccl_test(reduce)
add_rccl_test(scatter)
add_rccl_test(sendrecv)
add_rccl_test(all_reduce_bias)

if (ENABLE_ROCSHMEM_GIN AND RCCL_SOURCE_DIR)
    add_executable(gin_rocshmem_gda_factory_test ${CMAKE_CURRENT_SOURCE_DIR}/gin_rocshmem_gda_factory_test.cu)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gin_rocshmem_gda_factory_test.cu PROPERTIES LANGUAGE CXX)
    target_include_directories(gin_rocshmem_gda_factory_test PRIVATE
        ${ROCM_PATH}/include
        ${RCCL_SOURCE_DIR}/src/include
    )
    target_compile_options(gin_rocshmem_gda_factory_test PRIVATE -xhip)
    target_link_libraries(gin_rocshmem_gda_factory_test PRIVATE
        roc::rccl
        $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>)
    set_target_properties(gin_rocshmem_gda_factory_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
