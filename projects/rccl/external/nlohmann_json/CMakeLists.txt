# nlohmann/json 3.12.0 single-header — https://github.com/nlohmann/json/tree/v3.12.0
#
# Provides nlohmann_json::nlohmann_json. RCCL always builds against this copy, so it
# needs neither a system nlohmann_json package nor a FetchContent download.

if(TARGET nlohmann_json::nlohmann_json)
  return()
endif()

add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

target_include_directories(nlohmann_json INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(nlohmann_json INTERFACE cxx_std_11)
