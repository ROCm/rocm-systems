add_library(runtime_api_headers INTERFACE)
target_include_directories(runtime_api_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_library(ROCm::runtime_headers ALIAS runtime_api_headers)
