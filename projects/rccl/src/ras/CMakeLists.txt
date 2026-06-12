# RAS sources
set(RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
)

# >>> RCCL-DELTA: client.cc is NCCL's RAS_BINSRCFILES (the rcclras executable source); RCCL routes
# it through the hipify pipeline so the executable is built from the hipified copy. ras_internal.h
# is listed so the hipify pipeline copies/converts it into HIPIFY_DIR.
list(APPEND RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_internal.h
)
# <<< RCCL-DELTA

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
