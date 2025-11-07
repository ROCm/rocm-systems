#
#
#

rocprofiler_deactivate_clang_tidy()

add_executable(codeobj-library-tests)

set(CODEOBJ_LIB_TEST_SOURCES "codeobj_library_test.cpp")

target_sources(codeobj-library-tests PRIVATE ${CODEOBJ_LIB_TEST_SOURCES})

target_compile_definitions(codeobj-library-tests
                           PRIVATE -DCODEOBJ_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\")

configure_file(smallkernel.bin smallkernel.bin COPYONLY)
configure_file(hipcc_output.s hipcc_output.s COPYONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/smallkernel.bin
          ${CMAKE_CURRENT_BINARY_DIR}/hipcc_output.s
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin
    COMPONENT tests)

target_link_libraries(
    codeobj-library-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-glog
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            GTest::gtest
            GTest::gtest_main
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk::rocprofiler-sdk-elf)

rocprofiler_add_unit_test(TARGET codeobj-library-tests
                          SOURCES ${CODEOBJ_LIB_TEST_SOURCES})
