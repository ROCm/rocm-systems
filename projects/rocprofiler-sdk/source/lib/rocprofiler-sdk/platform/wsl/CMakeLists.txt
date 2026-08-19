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
        WARNING
            "WSL/DXG agent enumeration is NOT built into this rocprofiler-sdk: the hsakmt headers in ${ROCM_PATH} do not declare what the DXG topology read needs (the adapter LUID pair and HsaStructureSizes). Every released ROCm package ships headers in this state, 7.x included, so this is the expected outcome of a stock build and not a sign of a broken one. To get WSL support compiled in, install hsakmt headers built from this repository (projects/rocr-runtime/libhsakmt/include) into ${ROCM_PATH} and reconfigure."
        )
    set(ROCPROFILER_PLATFORM_WSL_HEADERS agent.hpp)
    set(ROCPROFILER_PLATFORM_WSL_SOURCES agent_unsupported.cpp)
endif()

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PLATFORM_WSL_SOURCES}
                                                      ${ROCPROFILER_PLATFORM_WSL_HEADERS})
