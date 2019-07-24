#
# Build dynamic Library object
#
set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/core/roctracer.cpp
  ${LIB_DIR}/proxy/proxy_queue.cpp
  ${LIB_DIR}/proxy/simple_proxy_queue.cpp
  ${LIB_DIR}/proxy/intercept_queue.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC} )
target_include_directories ( ${TARGET_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${HSA_RUNTIME_HSA_INC_PATH} ${HIP_INC_DIR} ${HCC_INC_DIR} )
target_link_libraries( ${TARGET_LIB} PRIVATE ${HSA_RUNTIME_LIB} c stdc++ )
execute_process ( COMMAND sh -xc "${ROOT_DIR}/script/hsaap.py ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH}" )

set ( KFD_LIB "kfdwrapper64" )
set ( KFD_LIB_SRC
  ${LIB_DIR}/kfd/kfd_wrapper.cpp
)
add_library ( ${KFD_LIB} SHARED ${KFD_LIB_SRC} )
target_include_directories ( ${KFD_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${HSA_RUNTIME_HSA_INC_PATH} )
target_link_libraries( ${KFD_LIB} PRIVATE c stdc++ )
add_custom_command (
  OUTPUT "${ROOT_DIR}/src/kfd/kfd_wrapper.cpp"
  COMMAND sh -xc "${ROOT_DIR}/script/kfdap.py ${ROOT_DIR} ${HSA_KMT_INC_PATH}"
  DEPENDS ${ROOT_DIR}/script/kfdap.py
)

set ( ROCTX_LIB "roctx64" )
set ( ROCTX_LIB_SRC
  ${LIB_DIR}/roctx/roctx.cpp
)
add_library ( ${ROCTX_LIB} SHARED ${ROCTX_LIB_SRC} )
target_include_directories ( ${ROCTX_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${HSA_RUNTIME_HSA_INC_PATH} )
target_link_libraries( ${ROCTX_LIB} PRIVATE c stdc++ )
