find_library(ROCM_SMI_LIB rocm_smi64 HINTS ${ROCM_PATH}/lib)

if(ROCM_SMI_LIB)
    set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
    find_package(hip CONFIG REQUIRED PATHS "${ROCM_PATH}")
    enable_language(HIP)

    set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests-v2/microbenchmarks)
    file(GLOB TEST_SRC_FILE ${TEST_DIR}/*.cpp)

    set_source_files_properties(${TEST_SRC_FILE} PROPERTIES LANGUAGE HIP)
    add_executable(pcie_bw_test ${TEST_SRC_FILE})
    target_link_libraries(pcie_bw_test PRIVATE hip::device)
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
