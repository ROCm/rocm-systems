# Add tests subdirectory
if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

# PMC CPU Collector Sources
set(pmc_cpu_sources
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device.hpp
)

# Add to parent variable (will be used by collectors CMakeLists.txt)
set(pmc_cpu_sources ${pmc_cpu_sources} PARENT_SCOPE)
