# Transport sources (paths relative to PROJECT_SOURCE_DIR to match SRC_FILES convention)
set(NET_IB_CAST_SOURCES
    #src/transport/net_ib_limits.h
    ${CMAKE_CURRENT_SOURCE_DIR}/net_ib_cast_inspect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/net_ib_fault_inject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/connect_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery_ainic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery_ainic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin.cc
)

# The cast headers (common_cast.h, p2p_cast.h, etc.) live in this directory.
# After hipify the copies land under HIPIFY_DIR/src/transport/net_ib_cast, so
# we pass that path via NET_IB_CAST_INCLUDE_DIRS to the parent scope where it
# is forwarded to target_include_directories(rccl ...).
set(NET_IB_CAST_INCLUDE_DIRS
    "${HIPIFY_SRC}/transport/net_ib_cast"
    "${HIPIFY_SRC}/transport"
    PARENT_SCOPE)

# Propagate sources to parent scope so src/CMakeLists.txt can use ${NET_IB_CAST_SOURCES}
set(NET_IB_CAST_SOURCES ${NET_IB_CAST_SOURCES} PARENT_SCOPE)
