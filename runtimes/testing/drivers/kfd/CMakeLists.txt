# SPDX-License-Identifier: MIT

enable_language(C)
add_library(kfd_test_uapi INTERFACE)
target_include_directories(
    kfd_test_uapi
    SYSTEM
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/uapi/include"
)
add_executable(kfd-version-contract version_contract.c)
target_link_libraries(kfd-version-contract PRIVATE kfd_test_uapi)
runtime_test_native(kfd-version-contract kfd)
runtime_test_register_suite(kfd kfd kfd-version-contract kfd.version-contract FALSE)
