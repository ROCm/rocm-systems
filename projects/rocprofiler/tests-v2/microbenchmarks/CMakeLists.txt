find_library(ROCM_SMI_LIB rocm_smi64 HINTS ${ROCM_PATH}/lib)

if(ROCM_SMI_LIB)
    # Set the HIP language runtime link flags as FindHIP does not set them.
    set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP
        ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
    set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
    set(CMAKE_HIP_ARCHITECTURES OFF)
    if(DEFINED ROCM_PATH)
        set(HIP_ROOT_DIR "${ROCM_PATH}/bin")
    endif()
    find_package(HIP REQUIRED MODULE)

    set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests-v2/microbenchmarks)
    file(GLOB TEST_SRC_FILE ${TEST_DIR}/*.cpp)

    set_source_files_properties(${TEST_SRC_FILE} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(pcie_bw_test ${TEST_SRC_FILE})
    add_test(
        NAME pcie_bw_test
        COMMAND $<TARGET_FILE:pcie_bw_test>
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
    set_tests_properties(
        pcie_bw_test
        PROPERTIES LABELS "v2;benchmarks" ENVIRONMENT
                   "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}" SKIP_REGULAR_EXPRESSION
                   "SIGBUS error. Aborting test" DISABLED TRUE)

    target_link_libraries(pcie_bw_test PRIVATE ${ROCM_SMI_LIB})
    target_link_options(pcie_bw_test PRIVATE "-Wl,--build-id=md5")
    set_target_properties(
        pcie_bw_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                "${PROJECT_BINARY_DIR}/tests-v2/microbenchmarks")
    install(
        TARGETS pcie_bw_test
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/microbenchmarks
            COMPONENT tests)
endif()
