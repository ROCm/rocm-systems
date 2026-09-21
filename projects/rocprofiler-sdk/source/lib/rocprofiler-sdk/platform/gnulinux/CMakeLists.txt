#
# Linux KFD sysfs agent enumeration. Linked unconditionally on Linux builds
# (selected at runtime when /sys/.../kfd/topology/nodes is present).
#

set(ROCPROFILER_PLATFORM_GNULINUX_HEADERS agent.hpp)
set(ROCPROFILER_PLATFORM_GNULINUX_SOURCES agent.cpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_GNULINUX_SOURCES}
                                           ${ROCPROFILER_PLATFORM_GNULINUX_HEADERS})
