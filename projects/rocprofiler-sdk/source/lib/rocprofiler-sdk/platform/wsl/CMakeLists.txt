#
# WSL/DXCore agent enumeration via libdxcore.so. Linked unconditionally on
# Linux builds since WSL is a Linux runtime; runtime selection picks this
# path only when /dev/dxg is present and libdxcore.so loads.
#

set(ROCPROFILER_PLATFORM_WSL_HEADERS agent.hpp)
set(ROCPROFILER_PLATFORM_WSL_SOURCES agent.cpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_WSL_SOURCES}
                                                      ${ROCPROFILER_PLATFORM_WSL_HEADERS})
