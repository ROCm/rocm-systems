set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(rocprofiler-perfetto-sdk STATIC sdk/perfetto.h sdk/perfetto.cc)
add_library(rocprofiler::perfetto-sdk ALIAS rocprofiler-perfetto-sdk)

target_include_directories(rocprofiler-perfetto-sdk
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/sdk>)
