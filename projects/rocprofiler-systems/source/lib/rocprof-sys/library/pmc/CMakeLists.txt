# PMC (Performance Metric Counter) Sampler Library

# Add subdirectories
add_subdirectory(common)
add_subdirectory(cpu)
if(ROCPROFSYS_USE_ROCM)
    add_subdirectory(gpu)
endif()
add_subdirectory(device_providers)
add_subdirectory(output_policies)
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
    ${pmc_cpu_types_sources}
    ${pmc_gpu_types_sources}
    ${pmc_device_provider_sources}
    ${pmc_output_policies_sources}
    ${pmc_collectors_sources}
)

# Create PMC library object
add_library(rocprofiler-systems-pmc-library OBJECT ${pmc_sources})

target_link_libraries(
    rocprofiler-systems-pmc-library
    PRIVATE
        rocprofiler-systems-core-library
        rocprofiler-systems::rocprofiler-systems-interface-library
)
