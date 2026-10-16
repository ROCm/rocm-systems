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
# loader) and NOT librdc, so it pulls in no amdsmi. librdc runs amdsmi_init in a
# load-time static initializer that throws when no GPU/driver is present, which
# would abort the process before main(); keeping these tests out of that library
# lets them run on GPU-less CI and satisfy the unit-test gate anywhere. Uses
# gtest_main since it needs no custom setup.
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
# RDC_KERNELS_HSACO_DIR points at the in-tree kernel sources; the test skips when
# that tree is not reachable (for example a packaged test run).
add_executable(rdc_diag_kernel_tests ${SRC_DIR}/test_diag_kernels.cc)
target_compile_definitions(
    rdc_diag_kernel_tests
    PRIVATE RDC_KERNELS_HSACO_DIR="${PROJECT_SOURCE_DIR}/rdc_libs/rdc_modules/kernels/hsaco"
)
target_link_libraries(rdc_diag_kernel_tests PRIVATE GTest::gtest_main pthread)
add_test(NAME rdc_diag_kernel_tests COMMAND rdc_diag_kernel_tests)

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
# replies. Compiles RdcTopologyLinkImpl.cc into the test, which defines the
# amdsmi calls that file makes, so it links neither librdc nor amd_smi and runs
# on GPU-less CI. librdc_bootstrap provides the logger.
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
    TARGETS ${RDC_CODEC_TESTS} ${RDC_UNIT_TESTS} rdc_diag_kernel_tests
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
