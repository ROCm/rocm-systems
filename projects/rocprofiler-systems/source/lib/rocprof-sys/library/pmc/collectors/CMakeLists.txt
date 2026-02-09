# PMC Collectors

# Add subdirectories
add_subdirectory(common)
if(ROCPROFSYS_USE_ROCM)
    add_subdirectory(gpu)
endif()

# Collect all sources
set(pmc_collectors_sources ${pmc_common_sources} ${pmc_gpu_sources})

# Add to parent variable
set(pmc_collectors_sources ${pmc_collectors_sources} PARENT_SCOPE)
