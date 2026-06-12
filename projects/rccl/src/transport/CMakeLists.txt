# Transport sources
set(TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nvls.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/generic.cc
)

# >>> RCCL-DELTA: RCCL-only header in src/transport (listed so hipify copies it into HIPIFY_DIR).
list(APPEND TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/net_ib_limits.h
)
# <<< RCCL-DELTA

add_subdirectory(net_ib)
# >>> RCCL-DELTA: net_ib_cast is an RCCL-only transport (NCCL has no net_ib_cast). It populates
# NET_IB_CAST_SOURCES / NET_IB_CAST_INCLUDE_DIRS in THIS scope; relay them one more level up to
# src/ scope (where src/CMakeLists.txt aggregates them and configures rccl include dirs).
add_subdirectory(net_ib_cast)
set(NET_IB_CAST_SOURCES ${NET_IB_CAST_SOURCES} PARENT_SCOPE)
set(NET_IB_CAST_INCLUDE_DIRS ${NET_IB_CAST_INCLUDE_DIRS} PARENT_SCOPE)
# <<< RCCL-DELTA

# Add transport sources to parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
