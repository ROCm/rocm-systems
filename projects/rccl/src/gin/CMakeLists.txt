# Gin sources
set(GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_proxy.cc
)

if(ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN)
    list(APPEND GIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_rocshmem_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_rocshmem_api.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_rocshmem_gda.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_rocshmem_gda.cc)
endif()

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)
