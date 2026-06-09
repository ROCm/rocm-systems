# RAS sources
set(RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
)

# >>> RCCL-DELTA: RCCL compiles the RAS client into librccl as well, and needs it hipified
#     so the rcclras executable (built in src/CMakeLists.txt) has a staged source. In NCCL
#     client.cc is built only as the standalone ncclras binary, not part of the library.
list(APPEND RAS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/client.cc)
# <<< RCCL-DELTA

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
