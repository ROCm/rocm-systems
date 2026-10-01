add_library(rocm_runtime_api_headers INTERFACE)
target_include_directories(
    rocm_runtime_api_headers
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
add_library(rocm_runtime::runtime_headers ALIAS rocm_runtime_api_headers)
