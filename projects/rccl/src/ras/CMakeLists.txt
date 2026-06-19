# RAS sources
set(RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
)

# >>> RCCL-DELTA
list(APPEND RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_internal.h
)
# <<< RCCL-DELTA

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
