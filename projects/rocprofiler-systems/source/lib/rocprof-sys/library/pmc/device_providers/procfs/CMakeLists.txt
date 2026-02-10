# Add subdirectories
add_subdirectory(drivers)

# Procfs Device Provider Sources
set(pmc_procfs_provider_sources
    ${CMAKE_CURRENT_LIST_DIR}/provider.hpp
    ${pmc_procfs_driver_sources}
)

# Add to parent variable (will be used by device_providers CMakeLists.txt)
set(pmc_procfs_provider_sources ${pmc_procfs_provider_sources} PARENT_SCOPE)
