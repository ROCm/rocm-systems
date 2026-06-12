# Transport sources
set(NET_IB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin.cc
)

# >>> RCCL-DELTA: RCCL-only headers in src/transport/net_ib (listed so the hipify pipeline copies/
# converts them into HIPIFY_DIR; the net_ib basenames common/gin/p2p are symlinked post-hipify by
# src/CMakeLists.txt). gdaki/gin_host_gdaki.h is included here because RCCL does not wire the gdaki
# subdirectory (see fenced add_subdirectory below).
list(APPEND NET_IB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gdaki/gin_host_gdaki.h
)
# <<< RCCL-DELTA

# >>> RCCL-DELTA: NCCL wires the gdaki (DOCA GPUNetIO) subdirectory here, which adds DOCA sources
# and gin_host_gdaki.cc to TRANSPORT_SOURCES. Those are CUDA/DOCA-only and are not built by RCCL,
# so the subdirectory is not added. The gdaki/CMakeLists.txt is kept as an NCCL mirror.
# | add_subdirectory(gdaki)
# <<< RCCL-DELTA

# Add transport sources to parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} ${NET_IB_SOURCES} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
