# Transport sources (paths relative to PROJECT_SOURCE_DIR to match SRC_FILES convention)
set(NET_IB_CAST_SOURCES
    src/transport/net_ib_cast/common_cast.h
    src/transport/net_ib_cast/connect_cast.h
    src/transport/net_ib_cast/gin_cast.h
    src/transport/net_ib_cast/p2p_cast.h
    src/transport/net_ib_cast/p2p_resiliency_cast.h
    src/transport/net_ib_cast/p2p_resiliency_recovery_cast.h
    src/transport/net_ib_cast/common.cc
    src/transport/net_ib_cast/connect.cc
    src/transport/net_ib_cast/gdr.cc
    src/transport/net_ib_cast/init.cc
    src/transport/net_ib_cast/p2p.cc
    src/transport/net_ib_cast/p2p_resiliency.cc
    src/transport/net_ib_cast/p2p_resiliency_recovery.cc
    src/transport/net_ib_cast/reg.cc
    src/transport/net_ib_cast/scheduler.cc
    # gin.cc and gin_host_gdaki.cc require gin/gin_host.h (CUDA/NCCL-only) — exclude from ROCm build.
    # src/transport/net_ib_cast/gin.cc
    # src/transport/net_ib_cast/gdaki/gin_host_gdaki.cc
)

# The cast headers (common_cast.h, p2p_cast.h, etc.) live in this directory.
# After hipify the copies land under HIPIFY_DIR/src/transport/net_ib_cast, so
# we pass that path via NET_IB_CAST_INCLUDE_DIRS to the parent scope where it
# is forwarded to target_include_directories(rccl ...).
set(NET_IB_CAST_INCLUDE_DIRS
    "${CMAKE_BINARY_DIR}/hipify/src/transport/net_ib_cast"
    PARENT_SCOPE)

# Propagate sources to parent scope so src/CMakeLists.txt can use ${NET_IB_CAST_SOURCES}
set(NET_IB_CAST_SOURCES ${NET_IB_CAST_SOURCES} PARENT_SCOPE)
