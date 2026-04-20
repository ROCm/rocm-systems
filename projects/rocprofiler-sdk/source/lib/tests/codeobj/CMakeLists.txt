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

# Compile a HIP kernel with debug info to test DWARF inline annotation. Two-step: compile
# to offload bundle, then unbundle to get the GPU ELF.
find_program(
    CODEOBJ_AMDCLANGPP REQUIRED
    NAMES amdclang++
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

find_program(
    CODEOBJ_OFFLOAD_BUNDLER REQUIRED
    NAMES clang-offload-bundler
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

# The arch only matters for compilation, not execution — the test only parses DWARF debug
# info from the resulting ELF.  Any valid arch works here.
set(_SYNCKERNEL_ARCH "gfx90a")
set(_SYNCKERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/syncthreads_kernel.hip")
set(_SYNCKERNEL_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/syncthreads_kernel_bundle.o")
set(_SYNCKERNEL_BIN "${CMAKE_CURRENT_BINARY_DIR}/syncthreads_kernel.bin")

add_custom_command(
    OUTPUT "${_SYNCKERNEL_BIN}"
    COMMAND
        ${CODEOBJ_AMDCLANGPP} -x hip -ggdb --offload-arch=${_SYNCKERNEL_ARCH}
        --offload-device-only -c -fno-gpu-rdc --no-offload-new-driver -o
        "${_SYNCKERNEL_BUNDLE}" "${_SYNCKERNEL_SRC}"
    COMMAND
        ${CODEOBJ_OFFLOAD_BUNDLER} --type=o --input=${_SYNCKERNEL_BUNDLE}
        --output=${_SYNCKERNEL_BIN} --unbundle
        --targets=hipv4-amdgcn-amd-amdhsa--${_SYNCKERNEL_ARCH}
    DEPENDS "${_SYNCKERNEL_SRC}"
    COMMENT
        "Compiling syncthreads_kernel.hip -> syncthreads_kernel.bin (${_SYNCKERNEL_ARCH})"
    VERBATIM)

add_custom_target(syncthreads_kernel_bin DEPENDS "${_SYNCKERNEL_BIN}")
add_dependencies(codeobj-library-tests syncthreads_kernel_bin)

install(
    FILES "${_SYNCKERNEL_BIN}"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests/unit-tests/bin
    COMPONENT tests
    OPTIONAL)

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
            rocprofiler-sdk::rocprofiler-sdk-abseil
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
