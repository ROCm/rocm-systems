# PMC CPU Data Types

set(pmc_cpu_types_sources
    ${CMAKE_CURRENT_LIST_DIR}/types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample.hpp
)

# Add to parent variable (will be used by pmc CMakeLists.txt)
set(pmc_cpu_types_sources ${pmc_cpu_types_sources} PARENT_SCOPE)
