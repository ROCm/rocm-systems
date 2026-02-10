# Procfs Driver Sources
set(pmc_procfs_driver_sources ${CMAKE_CURRENT_LIST_DIR}/driver.hpp)

# Add to parent variable (will be used by procfs provider CMakeLists.txt)
set(pmc_procfs_driver_sources ${pmc_procfs_driver_sources} PARENT_SCOPE)
