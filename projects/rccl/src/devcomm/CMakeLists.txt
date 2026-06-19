# DEVCOMM sources
set(DEVCOMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/devcomm_v22902.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/devcomm_v22907.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/devcomm_v23000.cc
)

# >>> RCCL-DELTA
list(APPEND DEVCOMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/devcomm_v22902.h
)
# <<< RCCL-DELTA

# Propagate sources to parent scope so src/CMakeLists.txt can use ${DEVCOMM_SOURCES}.
set(DEVCOMM_SOURCES ${DEVCOMM_SOURCES} PARENT_SCOPE)
