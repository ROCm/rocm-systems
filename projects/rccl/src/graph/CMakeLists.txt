# Graph sources
set(GRAPH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/topo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuning.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/xml.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/search.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/paths.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rings.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/trees.cc
)

# >>> RCCL-DELTA: RCCL-only graph sources (Rome model topology detection / consensus).
list(APPEND GRAPH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_models.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_topo_consensus.cc
)
# <<< RCCL-DELTA

# Add graph sources to parent scope
set(GRAPH_SOURCES ${GRAPH_SOURCES} PARENT_SCOPE)
