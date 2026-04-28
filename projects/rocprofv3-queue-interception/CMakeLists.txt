cmake_minimum_required(VERSION 3.21)
project(rocprofv3-qi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# HSA headers ship inside this same repo (rocr-runtime is a sibling project).
set(HSA_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc")

# Reuse trace_buffer implementation from sibling rocprofv3-min project.
set(ROCPROFV3_MIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rocprofv3-min")

# HIP headers + install destination come from a HIP install/stage directory.
# Override on the command line if your stage lives elsewhere:
#   cmake .. -DROCPROFV3_QI_STAGE_DIR=C:/path/to/stage
set(ROCPROFV3_QI_STAGE_DIR "C:/Users/gobhardw/workarea/full-workflow/stage"
    CACHE PATH "Path to the HIP install/stage directory (contains include/ and bin/)")
set(HIP_INC   "${ROCPROFV3_QI_STAGE_DIR}/include")
set(STAGE_DIR "${ROCPROFV3_QI_STAGE_DIR}")
set(STAGE_BIN "${STAGE_DIR}/bin")
set(STAGE_LIB "${STAGE_DIR}/lib")

# ---- DLL ----
add_library(rocprofv3-qi SHARED
    src/rocprofv3_qi.cpp
    src/hsa_wrappers.cpp
    ${ROCPROFV3_MIN_DIR}/src/trace_buffer.cpp
)
target_include_directories(rocprofv3-qi PRIVATE
    ${HSA_INC}
    ${HIP_INC}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ROCPROFV3_MIN_DIR}/src
)
target_compile_definitions(rocprofv3-qi PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    __HIP_PLATFORM_AMD__
    AMD_INTERNAL_BUILD
)
target_compile_options(rocprofv3-qi PRIVATE /EHsc /W3 /wd4244 /wd4267)

# ---- install ----
install(TARGETS rocprofv3-qi
        RUNTIME DESTINATION ${STAGE_BIN}
        LIBRARY DESTINATION ${STAGE_BIN}
        ARCHIVE DESTINATION ${STAGE_LIB})
