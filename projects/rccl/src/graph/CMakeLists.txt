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

# >>> RCCL-DELTA: RCCL-only sources/headers in src/graph (rome_* topology models are AMD
# additions; the *.h are listed so the hipify pipeline copies/converts them into HIPIFY_DIR).
list(APPEND GRAPH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_models.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_topo_consensus.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_models.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rome_topo_consensus.h
    ${CMAKE_CURRENT_SOURCE_DIR}/topo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/xml.h
)
# <<< RCCL-DELTA

# Add graph sources to parent scope
set(GRAPH_SOURCES ${GRAPH_SOURCES} PARENT_SCOPE)
