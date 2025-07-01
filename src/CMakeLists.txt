#
# Build dynamic Library object
#
set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/core/aql_profile.cpp
  ${LIB_DIR}/core/counters.cpp
  ${LIB_DIR}/core/threadtrace.cpp
  ${LIB_DIR}/core/spm_data.cpp
  ${LIB_DIR}/core/populate_aql.cpp
  ${LIB_DIR}/core/memorymanager.cpp
  ${LIB_DIR}/core/pm4_factory.cpp
  ${LIB_DIR}/core/gfx9_factory.cpp
  ${LIB_DIR}/core/gfx908_factory.cpp
  ${LIB_DIR}/core/gfx90a_factory.cpp
  ${LIB_DIR}/core/gfx940_factory.cpp
  ${LIB_DIR}/core/gfx10_factory.cpp
  ${LIB_DIR}/core/gfx11_factory.cpp
  ${LIB_DIR}/core/gfx12_factory.cpp
  ${LIB_DIR}/core/vega20_reg_init.cpp
  ${LIB_DIR}/core/parse_ip_discovery.cpp
  ${LIB_DIR}/core/navi_reg_init.cpp
  ${LIB_DIR}/core/ip_offset_table_init.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)

add_library ( ${TARGET_LIB} SHARED ${LIB_SRC} )
target_include_directories ( ${TARGET_LIB} PRIVATE ${LIB_DIR} ${API_PATH} ${LIB_DIR}/core/include)
target_link_libraries( ${TARGET_LIB} PRIVATE pthread hsa-runtime64::hsa-runtime64 )

## Generating definitions
set ( SCRIPT_DIR "${ROOT_DIR}/script" )
set ( GFXIP_DIR "${ROOT_DIR}/gfxip" )
set ( DEF_DIR "${ROOT_DIR}/src/def" )
set ( BINFO_TEMPL "${SCRIPT_DIR}/gpu_block_info.h" )
set ( BINFO_DEF "${GFXIP_DIR}/gpu_block_info.h" )
set ( BINFO_HEADER "${DEF_DIR}/gpu_block_info.h" )
set ( GFX9_TEMPL "${SCRIPT_DIR}/gfx9_def.h" )
set ( GFX9_DEF "${GFXIP_DIR}/gfx9_def.h" )
set ( GFX9_HEADER "${DEF_DIR}/gfx9_def.h" )
set ( GFX10_TEMPL "${SCRIPT_DIR}/gfx10_def.h" )
set ( GFX10_DEF "${GFXIP_DIR}/gfx10_def.h" )
set ( GFX10_HEADER "${DEF_DIR}/gfx10_def.h" )
set ( GFX11_TEMPL "${SCRIPT_DIR}/gfx11_def.h" )
set ( GFX11_DEF "${GFXIP_DIR}/gfx11_def.h" )
set ( GFX11_HEADER "${DEF_DIR}/gfx11_def.h" )
set ( GFX12_TEMPL "${SCRIPT_DIR}/gfx12_def.h" )
set ( GFX12_DEF "${GFXIP_DIR}/gfx12_def.h" )
set ( GFX12_HEADER "${DEF_DIR}/gfx12_def.h" )
include_directories ( ${ROOT_DIR} )
add_custom_target( mygenreset
  COMMAND sh -xc "cp ${BINFO_TEMPL} ${BINFO_HEADER}"
  COMMAND sh -xc "cp ${GFX9_TEMPL} ${GFX9_HEADER}"
  COMMAND sh -xc "cp ${GFX10_TEMPL} ${GFX10_HEADER}"
  COMMAND sh -xc "cp ${GFX11_TEMPL} ${GFX11_HEADER}"
  COMMAND sh -xc "cp ${GFX12_TEMPL} ${GFX12_HEADER}"
)
add_custom_target( mygen
  COMMAND sh -xc "cp ${BINFO_TEMPL} ${BINFO_HEADER}"
  COMMAND sh -xc "cp ${GFX9_TEMPL} ${GFX9_HEADER}"
  COMMAND sh -xc "cp ${GFX10_TEMPL} ${GFX10_HEADER}"
  COMMAND sh -xc "cp ${GFX11_TEMPL} ${GFX11_HEADER}"
  COMMAND sh -xc "cp ${GFX12_TEMPL} ${GFX12_HEADER}"
  COMMAND sh -xc "sed '/gfx9_def/ s/.*//' ${GFX9_TEMPL} >${GFX9_HEADER}"
  COMMAND sh -xc "${ROOT_DIR}/script/errextr.sh ${GFX9_HEADER} ${GFX9_DEF}"
  COMMAND sh -xc "sed '/gfx10_def/ s/.*//' ${GFX10_TEMPL} >${GFX10_HEADER}"
  COMMAND sh -xc "${ROOT_DIR}/script/errextr.sh ${GFX10_HEADER} ${GFX10_DEF}"
  COMMAND sh -xc "sed '/gfx11_def/ s/.*//' ${GFX11_TEMPL} >${GFX11_HEADER}"
  COMMAND sh -xc "${ROOT_DIR}/script/errextr.sh ${GFX11_HEADER} ${GFX11_DEF}"
  COMMAND sh -xc "sed '/gfx12_def/ s/.*//' ${GFX12_TEMPL} >${GFX12_HEADER}"
  COMMAND sh -xc "${ROOT_DIR}/script/errextr.sh ${GFX12_HEADER} ${GFX12_DEF}"
  COMMAND sh -xc "head -n1 ${BINFO_TEMPL} >${BINFO_HEADER}"
  COMMAND sh -xc "sed 's/_GPU_BLOCKINFO_H_/SRC_DEF_GPU_BLOCK_INFO_H_/' ${BINFO_DEF} >>${BINFO_HEADER}"
)


add_subdirectory(src/core)
if(AQLPROFILE_BUILD_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(src/util/tests)
endif()
