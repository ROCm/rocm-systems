# {fmt} 10.2.1 header-only — https://github.com/fmtlib/fmt/tree/10.2.1
#
# Provides fmt::fmt-header-only. RCCL always builds against this copy, so it
# needs neither a system fmt package nor a FetchContent download.

if(TARGET fmt::fmt-header-only)
  return()
endif()

add_library(fmt-header-only INTERFACE)
add_library(fmt::fmt-header-only ALIAS fmt-header-only)

target_include_directories(fmt-header-only INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(fmt-header-only INTERFACE FMT_HEADER_ONLY=1)
target_compile_features(fmt-header-only INTERFACE cxx_std_11)
