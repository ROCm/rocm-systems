#
#
#

rocprofiler_deactivate_clang_tidy()

add_executable(codeobj-library-tests)

set(CODEOBJ_LIB_TEST_SOURCES "codeobj_library_test.cpp")

target_sources(codeobj-library-tests PRIVATE ${CODEOBJ_LIB_TEST_SOURCES})

set(PKG_TEST_BIN_DIR "${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin")

target_compile_definitions(
    codeobj-library-tests
    PRIVATE CODEOBJ_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}/"
            CODEOBJ_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}/${PKG_TEST_BIN_DIR}/")

configure_file(smallkernel.bin smallkernel.bin COPYONLY)
configure_file(hipcc_output.s hipcc_output.s COPYONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/hipcc_output.s
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin
    COMPONENT tests)

find_program(
    ROCM_LLVM_STRIP
    NAMES llvm-strip
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES llvm/bin bin)

set(_SMALLBIN "${CMAKE_CURRENT_BINARY_DIR}/smallkernel.bin")
set(_SMALLBIN_STRIPPED "${CMAKE_CURRENT_BINARY_DIR}/smallkernel.bin.stripped")

if(ROCM_LLVM_STRIP)
    add_custom_command(
        OUTPUT "${_SMALLBIN_STRIPPED}"
        COMMAND "${ROCM_LLVM_STRIP}" -s "${_SMALLBIN}" -o "${_SMALLBIN_STRIPPED}"
        DEPENDS "${_SMALLBIN}"
        COMMENT
            "Stripping ALL symbols from smallkernel.bin so CPack won't try to create a dbgsym"
        VERBATIM)

    add_custom_target(strip_smallkernel ALL DEPENDS "${_SMALLBIN_STRIPPED}")

    install(
        FILES "${_SMALLBIN_STRIPPED}"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin"
        RENAME smallkernel.bin
        COMPONENT tests)
else()
    message(
        WARNING
            "llvm-strip not found; installing unmodified smallkernel.bin (CPack may fail on debuginfo)."
        )
    install(
        FILES "${_SMALLBIN}"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin"
        COMPONENT tests)
endif()

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

rocprofiler_add_unit_test(
    TARGET codeobj-library-tests
    SOURCES ${CODEOBJ_LIB_TEST_SOURCES}
    ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
