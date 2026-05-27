#
# Platform-specific agent enumeration. Each subdirectory contributes a
# self-contained namespace (rocprofiler::platform::{gnulinux,wsl,windows}).
# Selection happens at runtime in agent.cpp; Linux builds compile both
# gnulinux/ and wsl/ into the same shared library so the runtime can pick.
#

set(ROCPROFILER_PLATFORM_HEADERS agent.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_HEADERS})

add_subdirectory(gnulinux)
add_subdirectory(wsl)

if(WIN32)
    add_subdirectory(windows)
endif()
