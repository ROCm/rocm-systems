#
# WSL/DXCore agent enumeration via libdxcore.so. Linked unconditionally on
# Linux builds since WSL is a Linux runtime; runtime selection picks this
# path only when /dev/dxg is present and libdxcore.so loads.
#
# The enumerator reads the KMT topology through librocdxg, and reads it through
# hsakmt fields older headers do not declare. Every ROCm whose headers lack them
# ships no librocdxg either, so building against one compiles the enumerator out
# and leaves platform::wsl reporting itself unavailable - which is what it would
# report at run time on such a ROCm anyway, there being no thunk to load.
#

include(rocprofiler-sdk-utilities)

rocprofiler_sdk_dxg_topology_supported(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)

if(ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    set(ROCPROFILER_PLATFORM_WSL_HEADERS agent.hpp dxg_thunk.hpp dxg_topology.hpp)
    set(ROCPROFILER_PLATFORM_WSL_SOURCES agent.cpp dxg_topology.cpp)
else()
    message(
        STATUS
            "WSL/DXG agent enumeration disabled: the hsakmt headers in ${ROCM_PATH} do not declare everything the DXG topology read needs (adapter LUID, HsaStructureSizes, OverrideEngineId), and this ROCm ships no librocdxg to read through either."
        )
    set(ROCPROFILER_PLATFORM_WSL_HEADERS agent.hpp)
    set(ROCPROFILER_PLATFORM_WSL_SOURCES agent_unsupported.cpp)
endif()

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_WSL_SOURCES}
                                                      ${ROCPROFILER_PLATFORM_WSL_HEADERS})
