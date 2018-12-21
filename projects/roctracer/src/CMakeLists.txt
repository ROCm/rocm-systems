#
# Build dynamic Library object
#
set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/core/roctracer.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC} )
target_include_directories ( ${TARGET_LIB} PRIVATE ${LIB_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${HIP_INC_DIR} ${HCC_INC_DIR} )
target_link_libraries( ${TARGET_LIB} PRIVATE ${HSA_RUNTIME_LIB} c stdc++ )
execute_process ( COMMAND sh -xc "${ROOT_DIR}/script/hsaap.py ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH}" )
