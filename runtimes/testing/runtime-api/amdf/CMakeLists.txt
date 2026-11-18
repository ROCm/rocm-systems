# SPDX-License-Identifier: MIT

enable_language(C)
if(
    DEFINED ROCmRuntimeAPI_DIR
    AND NOT EXISTS "${ROCmRuntimeAPI_DIR}/ROCmRuntimeAPIConfig.cmake"
)
    message(
        FATAL_ERROR
        "Explicit ROCmRuntimeAPI_DIR does not contain ROCmRuntimeAPIConfig.cmake"
    )
endif()
find_package(ROCmRuntimeAPI CONFIG REQUIRED COMPONENTS AMDF)
add_executable(amdf-host-contract amdf_host_contract.c)
target_link_libraries(
    amdf-host-contract
    PRIVATE ROCmRuntimeAPI::AMDF ${CMAKE_DL_LIBS}
)
get_target_property(_headers ROCmRuntimeAPI::AMDF INTERFACE_INCLUDE_DIRECTORIES)
runtime_test_native(amdf-host-contract runtime-api "${_headers}")
runtime_test_register_suite(amdf-host runtime-api amdf-host-contract runtime-api.amdf.host-contract TRUE)
