# Mock AMDSMI library for AINIC development without hardware

add_library(rocprofiler-systems-amdsmi-mock STATIC amdsmi_mock.cpp amdsmi_mock.hpp)

target_include_directories(
    rocprofiler-systems-amdsmi-mock
    PUBLIC ${PROJECT_SOURCE_DIR}/source/lib
)

if(ROCPROFSYS_USE_ROCM)
    target_include_directories(rocprofiler-systems-amdsmi-mock PRIVATE /opt/rocm/include)
endif()

# No need to override ROCPROFSYS_USE_ROCM - use the project's setting
# The mock provides AINIC functions that work with the real AMDSMI

# Install headers
install(
    FILES amdsmi_mock.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-systems/core/mock
)

# Optionally build tests if testing is enabled
# if(ROCPROFSYS_BUILD_TESTING)
#     add_subdirectory(tests)
# endif()
