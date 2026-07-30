#
# Native Windows D3DKMT agent enumeration. Only added to the build when
# WIN32 is true (gated by the parent platform/CMakeLists.txt).
#

set(ROCPROFILER_PLATFORM_WINDOWS_HEADERS agent.hpp)
set(ROCPROFILER_PLATFORM_WINDOWS_SOURCES agent.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_WINDOWS_SOURCES}
                                           ${ROCPROFILER_PLATFORM_WINDOWS_HEADERS})
