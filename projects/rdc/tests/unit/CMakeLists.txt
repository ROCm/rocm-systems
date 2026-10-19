# Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC unit tests                        ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if(WIN32)
    message("rdc unit test suite is not supported on Windows platform")
    return()
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RDC_UNIT_TESTS "rdc_unit_tests")
set(RDC_CODEC_TESTS "rdc_codec_tests")

# Reuse a googletest checkout if the standalone rdctst suite already fetched one;
# otherwise fetch it here. This keeps the unit tests buildable with BUILD_TESTS=ON
# even when BUILD_STANDALONE=OFF (no grpc, no daemon).
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# rdc_codec_tests: pure-logic tests (entity codec, AFID field registration,
# library loader; the loader tests dlopen only libm and libc). Links only
# librdc_bootstrap (which holds the codec, the field descriptor table and the
# loader) and NOT librdc, so it pulls in no amdsmi and runs on GPU-less CI.
# Uses gtest_main since it needs no custom setup.
add_executable(
    ${RDC_CODEC_TESTS}
    ${SRC_DIR}/test_entity_codec.cc
    ${SRC_DIR}/test_afid_field.cc
    ${SRC_DIR}/test_library_loader.cc
)
target_include_directories(
    ${RDC_CODEC_TESTS}
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${RDC_CODEC_TESTS} PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(${RDC_CODEC_TESTS} PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME ${RDC_CODEC_TESTS} COMMAND ${RDC_CODEC_TESTS})

# rdc_diag_kernel_tests: packaging guard for the precompiled diagnostic HSACO
# kernels. Pure-logic and GPU-less: it reads the committed kernel assets
# directly, so it links no rdc libraries and no amdsmi and runs on GPU-less CI.
# It reads both roots that are there: RDC_KERNELS_HSACO_DIR, the build tree's
# kernel sources, and, for an installed copy, <prefix>/lib/rdc/hsaco, which it
# finds from its own path by the same step as the INSTALL_RPATH below
# ($ORIGIN/../../../lib). Where neither root is reachable the checks skip.
add_executable(rdc_diag_kernel_tests ${SRC_DIR}/test_diag_kernels.cc)
target_compile_definitions(
    rdc_diag_kernel_tests
    PRIVATE RDC_KERNELS_HSACO_DIR="${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/kernels/hsaco"
)
target_link_libraries(rdc_diag_kernel_tests PRIVATE GTest::gtest_main pthread)
set_target_properties(
    rdc_diag_kernel_tests
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
)
add_test(NAME rdc_diag_kernel_tests COMMAND rdc_diag_kernel_tests)

# rdc_topology_tests: rdc_device_topology_get() against fake AMD SMI calls that
# the test defines. It compiles RdcTopologyLinkImpl.cc instead of linking librdc,
# so it needs no GPU; librdc_bootstrap provides the logger.
add_executable(
    rdc_topology_tests
    ${SRC_DIR}/test_topology_link.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcTopologyLinkImpl.cc
)
target_include_directories(
    rdc_topology_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_topology_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(rdc_topology_tests PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME rdc_topology_tests COMMAND rdc_topology_tests)

# rdc_client_tests: standalone (client) mode as rdci uses it. rdc_connect()
# dlopens librdc_client.so (gRPC); needs no GPU and no rdcd. Built only when the
# client library is.
if(TARGET rdc_client)
    add_executable(rdc_client_tests ${SRC_DIR}/test_client_shutdown.cc)
    target_include_directories(rdc_client_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(
        rdc_client_tests
        PRIVATE rdc_bootstrap GTest::gtest_main pthread ${CMAKE_DL_LIBS}
    )
    add_dependencies(rdc_client_tests rdc_client)
    set_target_properties(
        rdc_client_tests
        PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
    )
    add_test(NAME rdc_client_tests COMMAND rdc_client_tests)
    # librdc_bootstrap dlopens librdc_client.so by name; find it in the build tree.
    set_tests_properties(
        rdc_client_tests
        PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rdc_client>"
    )
    install(
        TARGETS rdc_client_tests
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
        COMPONENT ${TESTS_COMPONENT}
    )
endif()

# rdc_link_status_tests: rdc_link_status_get() against fake amdsmi link-status
# replies, and both it and rdc_device_topology_get() with an unavailable GPU.
# Compiles RdcTopologyLinkImpl.cc into the test, which defines the amdsmi calls
# that file makes, so it links neither librdc nor amd_smi and runs on GPU-less
# CI. librdc_bootstrap provides the logger.
add_executable(
    rdc_link_status_tests
    ${SRC_DIR}/test_link_status.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcTopologyLinkImpl.cc
)
target_include_directories(
    rdc_link_status_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_link_status_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(
    rdc_link_status_tests
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
)
add_test(NAME rdc_link_status_tests COMMAND rdc_link_status_tests)
install(
    TARGETS rdc_link_status_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_power_usage_tests: RDC_FI_POWER_USAGE against fake amdsmi power and
# gpu_metrics replies. Compiles the metric fetcher into the test, which defines
# the amdsmi calls the tests reach, so it runs on GPU-less CI. amd_smi provides
# the calls the tests never make, librdc_bootstrap the logger.
add_executable(
    rdc_power_usage_tests
    ${SRC_DIR}/test_power_usage.cc
    ${PROJECT_SOURCE_DIR}/common/rdc_capabilities.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcMetricFetcherImpl.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/SmiUtils.cc
)
target_include_directories(
    rdc_power_usage_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(
    rdc_power_usage_tests
    PRIVATE rdc_bootstrap amd_smi cap GTest::gtest_main pthread
)
set_target_properties(
    rdc_power_usage_tests
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib:${AMD_SMI_LIB_DIR}"
)
add_test(NAME rdc_power_usage_tests COMMAND rdc_power_usage_tests)
install(
    TARGETS rdc_power_usage_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_smi_diag_tests: the AMD SMI checks of rdci diag (compute processes, node
# topology, GPU parameters) and the diagnostic module that runs them. Compiles
# RdcSmiDiagnosticImpl.cc and RdcDiagnosticModule.cc into the test, which defines
# the amdsmi calls they make and the two SmiUtils functions they call
# (get_processor_handle_from_id, Smi2RdcError), so it links neither librdc nor
# amd_smi and runs on GPU-less CI. librdc_bootstrap provides the logger.
add_executable(
    rdc_smi_diag_tests
    ${SRC_DIR}/test_smi_diag.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcSmiDiagnosticImpl.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcDiagnosticModule.cc
)
target_include_directories(
    rdc_smi_diag_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_smi_diag_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
# The compute process check leaves out the process that runs it: the test's
# getpid() returns an ID no fake process has (test_smi_diag.cc, kOwnPid). The
# wrap covers every object linked into this binary (GoogleTest's too when it is
# static); code here that needs the real ID would call __real_getpid().
target_link_options(rdc_smi_diag_tests PRIVATE "LINKER:--wrap=getpid")
set_target_properties(rdc_smi_diag_tests PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME rdc_smi_diag_tests COMMAND rdc_smi_diag_tests)
install(
    TARGETS rdc_smi_diag_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_unit_tests: hardware-aware SmiUtils tests. Links librdc (flat table,
# partition helpers) and amd_smi. Provides its own main() to init amdsmi once.
# Hardware-dependent tests self-skip when no AMD GPU is present.
add_executable(${RDC_UNIT_TESTS} ${SRC_DIR}/test_smi_utils.cc)
target_include_directories(
    ${RDC_UNIT_TESTS}
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(
    ${RDC_UNIT_TESTS}
    PRIVATE rdc rdc_bootstrap amd_smi GTest::gtest pthread
)

# Find the amd_smi shared library next to librdc at runtime.
set_target_properties(
    ${RDC_UNIT_TESTS}
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib:${AMD_SMI_LIB_DIR}"
)
add_test(NAME ${RDC_UNIT_TESTS} COMMAND ${RDC_UNIT_TESTS})

install(
    TARGETS ${RDC_CODEC_TESTS} ${RDC_UNIT_TESTS} rdc_diag_kernel_tests rdc_topology_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_gpu_table_tests: the flat GPU table and RDC's AMD SMI session against fake
# AMD SMI calls. Compiles SmiUtils.cc, so it needs neither librdc nor a GPU, and
# links no amd_smi: test_gpu_table.cc defines the amdsmi calls that file makes.
add_executable(
    rdc_gpu_table_tests
    ${SRC_DIR}/test_gpu_table.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/SmiUtils.cc
)
target_include_directories(
    rdc_gpu_table_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_gpu_table_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(rdc_gpu_table_tests PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME rdc_gpu_table_tests COMMAND rdc_gpu_table_tests)
install(
    TARGETS rdc_gpu_table_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_updater_tests: RdcMetricsUpdaterImpl's stop() and start() against a test
# watch table. Compiles RdcMetricsUpdaterImpl.cc, so it needs neither librdc nor
# a GPU.
add_executable(
    rdc_updater_tests
    ${SRC_DIR}/test_metrics_updater.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcMetricsUpdaterImpl.cc
)
target_include_directories(
    rdc_updater_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_updater_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(rdc_updater_tests PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME rdc_updater_tests COMMAND rdc_updater_tests)
install(
    TARGETS rdc_updater_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_status_tests: the statuses the health calls return for a GPU group that does
# not exist, and rdc_status_string(). It compiles RdcWatchTableImpl.cc and
# RdcGroupSettingsImpl.cc instead of linking librdc, so it needs no GPU;
# librdc_bootstrap provides the logger and rdc_status_string().
add_executable(
    rdc_status_tests
    ${SRC_DIR}/test_status_codes.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcWatchTableImpl.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcGroupSettingsImpl.cc
)
target_include_directories(
    rdc_status_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(rdc_status_tests PRIVATE rdc_bootstrap GTest::gtest_main pthread)
set_target_properties(rdc_status_tests PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib")
add_test(NAME rdc_status_tests COMMAND rdc_status_tests)
install(
    TARGETS rdc_status_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_partition_count_tests: RDC_FI_GPU_PARTITION_COUNT against fake amdsmi
# partition profiles. Compiles the metric fetcher into the test, which defines
# the amdsmi calls the tests reach, so it runs on GPU-less CI. amd_smi provides
# the calls the tests never make, librdc_bootstrap the logger.
add_executable(
    rdc_partition_count_tests
    ${SRC_DIR}/test_partition_count.cc
    ${PROJECT_SOURCE_DIR}/common/rdc_capabilities.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcMetricFetcherImpl.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/SmiUtils.cc
)
target_include_directories(
    rdc_partition_count_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(
    rdc_partition_count_tests
    PRIVATE rdc_bootstrap amd_smi cap GTest::gtest_main pthread
)
set_target_properties(
    rdc_partition_count_tests
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib:${AMD_SMI_LIB_DIR}"
)
add_test(NAME rdc_partition_count_tests COMMAND rdc_partition_count_tests)
install(
    TARGETS rdc_partition_count_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_metric_fetcher_tests: builds and destroys metric fetchers from several
# threads at once. Compiles the metric fetcher into the test, so it runs on
# GPU-less CI. amd_smi provides the calls the fetcher's other functions make,
# librdc_bootstrap the logger.
add_executable(
    rdc_metric_fetcher_tests
    ${SRC_DIR}/test_metric_fetcher.cc
    ${PROJECT_SOURCE_DIR}/common/rdc_capabilities.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcMetricFetcherImpl.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/SmiUtils.cc
)
target_include_directories(
    rdc_metric_fetcher_tests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${AMD_SMI_INCLUDE_DIR}
)
target_link_libraries(
    rdc_metric_fetcher_tests
    PRIVATE rdc_bootstrap amd_smi cap GTest::gtest_main pthread
)
set_target_properties(
    rdc_metric_fetcher_tests
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib:${AMD_SMI_LIB_DIR}"
)
add_test(NAME rdc_metric_fetcher_tests COMMAND rdc_metric_fetcher_tests)
set_tests_properties(rdc_metric_fetcher_tests PROPERTIES TIMEOUT 120)
install(
    TARGETS rdc_metric_fetcher_tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
    COMPONENT ${TESTS_COMPONENT}
)

# rdc_rocr_diag_tests: the ROCr checks of rdci diag (compute queue, system memory)
# against a link-time fake of the HSA functions the module calls (fake_hsa.cc).
# It compiles the module's own sources, so it needs neither ROCr nor a GPU and
# links no hsa-runtime64; librdc_bootstrap provides the logger. The module's
# sources and the fake need the HSA headers, which are looked for under ROCM_DIR
# and nowhere else: without them the target is not built and a marker is
# installed in its place, which tests/ci/test_gtest_runner.py skips on.
find_path(RDC_ROCR_TEST_HSA_H hsa/hsa.h PATHS ${ROCM_DIR}/include NO_DEFAULT_PATH)
find_path(RDC_ROCR_TEST_HSA_EXT_H hsa/hsa_ext_amd.h PATHS ${ROCM_DIR}/include NO_DEFAULT_PATH)
message(STATUS "RDC_ROCR_TEST_HSA_H=${RDC_ROCR_TEST_HSA_H}")
message(STATUS "RDC_ROCR_TEST_HSA_EXT_H=${RDC_ROCR_TEST_HSA_EXT_H}")

# The module defines its two test hooks only where RocrTestHooks.h is, which is
# F22's tree: in a tree without it the tests that need a hook are skipped.
if(EXISTS "${PROJECT_SOURCE_DIR}/include/rdc_modules/rdc_rocr/RocrTestHooks.h")
    set(RDC_ROCR_TEST_HOOKS 1)
else()
    set(RDC_ROCR_TEST_HOOKS 0)
endif()
message(STATUS "RDC_ROCR_TEST_HOOKS=${RDC_ROCR_TEST_HOOKS}")

# The module's own sources, which is the list of
# rdc_libs/rdc_modules/rdc_rocr/CMakeLists.txt without its RdcLogger.cc: the targets below link
# librdc_bootstrap, which carries the logger, so compiling it again here would give each of them
# a second copy of it.
set(RDC_ROCR_TEST_MODULE_SRC
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/RdcDiagnosticLib.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/TestBase.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/ComputeQueueTest.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/MemoryTest.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/MemoryAccess.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/base_rocr_utils.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/common.cc
    ${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/rdc_rocr/RdcRocrBase.cc
)

if(RDC_ROCR_TEST_HSA_H AND RDC_ROCR_TEST_HSA_EXT_H)
    add_executable(
        rdc_rocr_diag_tests
        ${SRC_DIR}/test_rocr_diag.cc
        ${SRC_DIR}/fake_hsa.cc
        ${RDC_ROCR_TEST_MODULE_SRC}
    )
    target_include_directories(
        rdc_rocr_diag_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${COMMON_DIR}
            ${AMD_SMI_INCLUDE_DIR}
            ${RDC_ROCR_TEST_HSA_H}
    )
    target_compile_definitions(
        rdc_rocr_diag_tests
        PRIVATE RDC_ROCR_TEST_HOOKS=${RDC_ROCR_TEST_HOOKS}
    )
    target_link_libraries(
        rdc_rocr_diag_tests
        PRIVATE rdc_bootstrap GTest::gtest_main pthread ${CMAKE_DL_LIBS}
    )
    set_target_properties(
        rdc_rocr_diag_tests
        PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
    )
    add_test(NAME rdc_rocr_diag_tests COMMAND rdc_rocr_diag_tests)
    install(
        TARGETS rdc_rocr_diag_tests
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
        COMPONENT ${TESTS_COMPONENT}
    )

    # rdc_rocr_pin_tests: the module as a shared library, which the test dlopens
    # and dlcloses as rdcd's loader does, so that the pin that keeps an abandoned
    # check's code mapped can be seen. Neither library is installed: the fake
    # exports an absolute rdc_zero_symbol, which only the loader test wants, and
    # the module library is built to be unmapped by a dlclose without the pin
    # (-fno-gnu-unique, so that neither library defines a unique symbol).
    add_library(rdc_rocr_fake_hsa SHARED ${SRC_DIR}/fake_hsa.cc)
    target_include_directories(
        rdc_rocr_fake_hsa
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${COMMON_DIR}
            ${AMD_SMI_INCLUDE_DIR}
            ${RDC_ROCR_TEST_HSA_H}
    )
    target_compile_definitions(rdc_rocr_fake_hsa PRIVATE RDC_ROCR_TEST_HOOKS=${RDC_ROCR_TEST_HOOKS})
    target_link_options(rdc_rocr_fake_hsa PRIVATE "LINKER:--defsym=rdc_zero_symbol=0")

    add_library(rdc_rocr_pin_module SHARED ${RDC_ROCR_TEST_MODULE_SRC})
    target_include_directories(
        rdc_rocr_pin_module
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${COMMON_DIR}
            ${AMD_SMI_INCLUDE_DIR}
            ${RDC_ROCR_TEST_HSA_H}
    )
    target_compile_definitions(
        rdc_rocr_pin_module
        PRIVATE RDC_ROCR_TEST_HOOKS=${RDC_ROCR_TEST_HOOKS}
    )
    target_link_libraries(
        rdc_rocr_pin_module
        PRIVATE rdc_rocr_fake_hsa rdc_bootstrap pthread ${CMAKE_DL_LIBS}
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(rdc_rocr_fake_hsa PRIVATE -fno-gnu-unique)
        target_compile_options(rdc_rocr_pin_module PRIVATE -fno-gnu-unique)
    endif()

    add_executable(rdc_rocr_pin_tests ${SRC_DIR}/test_rocr_pin.cc)
    target_include_directories(
        rdc_rocr_pin_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${COMMON_DIR}
            ${AMD_SMI_INCLUDE_DIR}
            ${RDC_ROCR_TEST_HSA_H}
    )
    target_compile_definitions(
        rdc_rocr_pin_tests
        PRIVATE
            RDC_ROCR_TEST_HOOKS=${RDC_ROCR_TEST_HOOKS}
            RDC_ROCR_FAKE_HSA_PATH="$<TARGET_FILE:rdc_rocr_fake_hsa>"
            RDC_ROCR_PIN_MODULE_PATH="$<TARGET_FILE:rdc_rocr_pin_module>"
    )
    target_link_libraries(
        rdc_rocr_pin_tests
        PRIVATE rdc_bootstrap GTest::gtest_main pthread ${CMAKE_DL_LIBS}
    )
    add_dependencies(rdc_rocr_pin_tests rdc_rocr_fake_hsa rdc_rocr_pin_module)
    add_test(NAME rdc_rocr_pin_tests COMMAND rdc_rocr_pin_tests)
else()
    # The exact negation of the guard above: the runner test skips the binary
    # only when this marker is installed and the binary is not.
    set(RDC_ROCR_TEST_MARKER ${CMAKE_CURRENT_BINARY_DIR}/rdc_rocr_diag_tests.absent)
    file(
        WRITE ${RDC_ROCR_TEST_MARKER}
        "rdc_rocr_diag_tests was not built: hsa/hsa.h=${RDC_ROCR_TEST_HSA_H}, "
        "hsa/hsa_ext_amd.h=${RDC_ROCR_TEST_HSA_EXT_H}, searched ${ROCM_DIR}/include\n"
    )
    install(
        FILES ${RDC_ROCR_TEST_MARKER}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests
        COMPONENT ${TESTS_COMPONENT}
    )
endif()
