# Gin sources
set(GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_proxy.cc
)

if(ENABLE_ROCSHMEM)
    list(APPEND GIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_rocshmem.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_rocshmem.cc)
endif()

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)
