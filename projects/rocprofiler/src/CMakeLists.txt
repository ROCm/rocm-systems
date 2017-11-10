#
# Build dynamic Library object
#
set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/core/rocprofiler.cpp
  ${LIB_DIR}/core/proxy_queue.cpp
  ${LIB_DIR}/core/simple_proxy_queue.cpp
  ${LIB_DIR}/core/intercept_queue.cpp
  ${LIB_DIR}/core/metrics.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC} )
target_include_directories ( ${TARGET_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${TARGET_LIB} PRIVATE ${HSA_RUNTIME_LIB} c stdc++)
