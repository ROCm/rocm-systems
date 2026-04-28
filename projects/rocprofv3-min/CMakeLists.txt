cmake_minimum_required(VERSION 3.21)
project(rocprofv3-min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# HSA headers ship inside this same repo (rocr-runtime is a sibling project).
set(HSA_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc")

# HIP headers + install destination come from a HIP install/stage directory.
# Override on the command line if your stage lives elsewhere:
#   cmake .. -DROCPROFV3_MIN_STAGE_DIR=C:/path/to/stage
set(ROCPROFV3_MIN_STAGE_DIR "C:/Users/gobhardw/workarea/full-workflow/stage"
    CACHE PATH "Path to the HIP install/stage directory (contains include/ and bin/)")
set(HIP_INC   "${ROCPROFV3_MIN_STAGE_DIR}/include")
set(STAGE_DIR "${ROCPROFV3_MIN_STAGE_DIR}")
set(STAGE_BIN "${STAGE_DIR}/bin")
set(STAGE_LIB "${STAGE_DIR}/lib")

# ---- DLL ----
add_library(rocprofv3-min SHARED
    src/rocprofv3_min.cpp
    src/hip_wrappers.cpp
    src/hsa_wrappers.cpp
    src/trace_buffer.cpp
)
target_include_directories(rocprofv3-min PRIVATE
    ${HSA_INC}
    ${HIP_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(rocprofv3-min PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    __HIP_PLATFORM_AMD__
    AMD_INTERNAL_BUILD
)
target_compile_options(rocprofv3-min PRIVATE /EHsc /W3 /wd4244 /wd4267)

# ---- launcher ----
add_executable(rocprofv3 src/rocprofv3.cpp)
target_compile_options(rocprofv3 PRIVATE /EHsc /W3)
target_compile_definitions(rocprofv3 PRIVATE
    NOMINMAX _CRT_SECURE_NO_WARNINGS)

# ---- install ----
install(TARGETS rocprofv3-min
        RUNTIME DESTINATION ${STAGE_BIN}
        LIBRARY DESTINATION ${STAGE_BIN}
        ARCHIVE DESTINATION ${STAGE_LIB})
install(TARGETS rocprofv3
        RUNTIME DESTINATION ${STAGE_BIN})
