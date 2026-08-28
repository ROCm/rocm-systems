# Gin sources
set(GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_proxy.cc
)

# >>> RCCL-DELTA: RCCL-only sources/headers in src/gin
# (the *.h are listed so the hipify pipeline copies/converts them into HIPIFY_DIR)
list(APPEND GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_gpucontext/gpucontext.h
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_gpucontext/gpucontext_v1.h
)
# <<< RCCL-DELTA

# >>> RCCL-DELTA: conditional rocSHMEM sources
if(ENABLE_ROCSHMEM_GIN)
  list(APPEND GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_rocshmem_gda.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_rocshmem_gda_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_plugin_anvil_sdma.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_anvil_ipc_table_host.cc
  )
endif()
# <<< RCCL-DELTA

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)