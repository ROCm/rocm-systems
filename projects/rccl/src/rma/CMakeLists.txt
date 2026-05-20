# RMA sources
set(RMA_SOURCES
    src/rma/rma.cc
    src/rma/rma_proxy.cc
    src/rma/rma_proxy_progress.cc
    src/rma/rma_proxy_launch.cc
)

# Add RMA sources to parent scope
set(RMA_SOURCES ${RMA_SOURCES} PARENT_SCOPE)
