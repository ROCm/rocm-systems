# ---------------------------------------------------------------------------
# src/graph/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_GRAPH_SIBLINGS
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/connect.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc)
set_source_files_properties(
  ${_MICRO_GRAPH_SIBLINGS}
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  PROPERTIES GENERATED TRUE)
# Real sibling units read their params through graph_fakes.cc's g_loadParam, as the unit under test does. Only in
# targets defining RCCL_MICRO_GRAPH_FAKES: other directories compile some of these files against their own fakes.
set(_micro_graph_fakes_on "$<IN_LIST:RCCL_MICRO_GRAPH_FAKES,$<TARGET_PROPERTY:COMPILE_DEFINITIONS>>")
set_property(SOURCE ${_MICRO_GRAPH_SIBLINGS} APPEND PROPERTY COMPILE_OPTIONS
  "$<${_micro_graph_fakes_on}:-include;fakes/graph_fakes.h;-include;fakes/param_redirect.h>")
unset(_micro_graph_fakes_on)
set(_MICRO_GRAPH_DEFS
  ${_MICRO_AMDSMI_DEF}
  CONNECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/connect.cc"
  PATHS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc"
  RCCL_GRAPH_GEN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rccl_graph_gen.cc"
  RINGS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc"
  ROME_MODELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc"
  SEARCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc"
  TOPO_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc"
  TREES_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc"
  XML_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc"
  TOPO_EXPL_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tools/topo_expl/models")

# One src/graph/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_graph_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} graph/${_test}
    OBJECT_PREFIX rccl-graph-obj
    COMMON_DEFS ${_MICRO_GRAPH_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_graph.yaml"
    ${ARGN})
endmacro()

# The graph_fakes.cc tests: real src/graph units, everything outside src/graph faked.
macro(rccl_add_graph_fakes_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} graph/${_test}
    OBJECT_PREFIX rccl-graph-fakes-obj
    COMMON_DEFS ${_MICRO_GRAPH_DEFS} RCCL_MICRO_GRAPH_FAKES
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_graph.yaml"
    ${ARGN})
endmacro()

# graph-dev-test.cc runs the real topology units against nccl_fakes.cc and os_fakes.cc instead.
rccl_add_graph_micro_binary(rccl-UnitTestsMicroGraphDev graph-dev-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  EXTRA_DEFS
    RCCL_NCCL_FAKES_OMIT_GRAPH
    RCCL_TRANSPORT_STUBS_OMIT_ncclPxnDisable)

rccl_add_graph_micro_binary(rccl-UnitTestsMicroGraphRcclGraphGen rccl-graph-gen-test.cc
  SOURCES
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_graph_micro_binary(rccl-UnitTestsMicroGraphTrees trees-test.cc)

# The graph_fakes.cc binaries share their objects, so they all drop ENABLE_WARP_SPEED: the GraphConnect cases
# assert the non-warp-speed channel caps (gfx1250, MNNVL), and that arm calls rccl_wrap.cc, which none link.
set(_micro_graph_saved_defs "${RCCL_COMMON_COMPILE_DEFS}")
list(REMOVE_ITEM RCCL_COMMON_COMPILE_DEFS ENABLE_WARP_SPEED)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphConnect graph-connect-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphPaths paths-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/connect.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphRings rings-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphRomeModels rome-models-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphSearch search-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    graph/topo_include.cc  # topo.cc, included (not compiled) for TopoSetNetDevsPolicy
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphTopo topo-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_graph_fakes_micro_binary(rccl-UnitTestsMicroGraphXml xml-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/graph_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

set(RCCL_COMMON_COMPILE_DEFS "${_micro_graph_saved_defs}")
unset(_micro_graph_saved_defs)
