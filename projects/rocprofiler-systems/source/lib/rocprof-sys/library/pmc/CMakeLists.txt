# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC (Performance Metric Counter) Sampler Library

# Add subdirectories
add_subdirectory(common)
add_subdirectory(device_providers)
add_subdirectory(collectors)

# Sampler API sources
set(pmc_sampler_sources
    ${CMAKE_CURRENT_LIST_DIR}/sampler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sampler.cpp
)

# Collect all PMC sources
set(pmc_sources
    ${pmc_sampler_sources}
    ${pmc_common_types_sources}
    ${pmc_device_provider_sources}
    ${pmc_collectors_sources}
)

# Create PMC library object
add_library(rocprofiler-systems-pmc-library OBJECT ${pmc_sources})

# Add AINIC support definition if available
# PUBLIC so that tests linking to this library also get the definition
if(ROCPROFSYS_BUILD_AINIC)
    target_compile_definitions(
        rocprofiler-systems-pmc-library
        PUBLIC ROCPROFSYS_BUILD_AINIC=1
    )
endif()

target_link_libraries(
    rocprofiler-systems-pmc-library
    PRIVATE
        rocprofiler-systems-core-library
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-rocm
)
