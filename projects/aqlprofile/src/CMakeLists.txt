#
# Build dynamic Library object
#
set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/core/aql_profile.cpp
  ${LIB_DIR}/core/counters.cpp
  ${LIB_DIR}/core/threadtrace.cpp
  ${LIB_DIR}/core/spm_data.cpp
  ${LIB_DIR}/core/spm_decode.cpp
  ${LIB_DIR}/core/spm_v2.cpp
  ${LIB_DIR}/core/populate_aql.cpp
  ${LIB_DIR}/core/memorymanager.cpp
  ${LIB_DIR}/core/pm4_factory.cpp
  ${LIB_DIR}/core/gfx9_factory.cpp
  ${LIB_DIR}/core/gfx908_factory.cpp
  ${LIB_DIR}/core/gfx90a_factory.cpp
  ${LIB_DIR}/core/gfx940_factory.cpp
  ${LIB_DIR}/core/gfx10_factory.cpp
  ${LIB_DIR}/core/gfx11_factory.cpp
  ${LIB_DIR}/core/gfx115x_factory.cpp
  ${LIB_DIR}/core/gfx12_factory.cpp
  ${LIB_DIR}/core/gfx1250_factory.cpp
  ${LIB_DIR}/core/vega20_reg_init.cpp
  ${LIB_DIR}/core/parse_ip_discovery.cpp
  ${LIB_DIR}/core/navi_reg_init.cpp
  ${LIB_DIR}/core/ip_offset_table_init.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)

add_library ( ${TARGET_LIB} SHARED ${LIB_SRC} )
add_library (aqlprofile::aqlprofile-lib ALIAS ${TARGET_LIB})
target_include_directories ( ${TARGET_LIB}
  PUBLIC  ${LIB_DIR}
  PRIVATE ${API_PATH}
          ${LIB_DIR}/core/include
)
if ( WIN32 )
  # Link the module-definition file so the linker exports all public symbols
  target_sources ( ${TARGET_LIB} PRIVATE ${LIB_DIR}/hsa-amd-aqlprofile64.def )
  # Windows HSA symbol source. By default the pre-built HIP import library
  # (amdhip64.lib) re-exports a subset of the HSA runtime. But amdhip64 embeds
  # its own private rocr instance, so a test that also loads a standalone
  # hsa-runtime64.dll ends up with two runtimes and cross-runtime handle
  # mismatches. Set AQLPROFILE_WIN_LINK_ROCR=ON (with HSA_RUNTIME64_LIB_DIR
  # pointing at a standalone hsa-runtime64.lib) to link the full rocr runtime
  # instead, so the whole stack shares a single HSA runtime.
  option ( AQLPROFILE_WIN_LINK_ROCR
           "Link the standalone rocr hsa-runtime64 import lib instead of amdhip64 on Windows" OFF )
  if ( AQLPROFILE_WIN_LINK_ROCR )
    find_library ( HSA_RUNTIME64_LIB
      NAMES hsa-runtime64
      HINTS "${HSA_RUNTIME64_LIB_DIR}" "${CMAKE_INSTALL_PREFIX}/lib" )
    if ( NOT HSA_RUNTIME64_LIB OR HSA_RUNTIME64_LIB STREQUAL "HSA_RUNTIME64_LIB-NOTFOUND" )
      message ( FATAL_ERROR "AQLPROFILE_WIN_LINK_ROCR=ON but hsa-runtime64.lib not found. "
        "Set HSA_RUNTIME64_LIB_DIR to the directory containing it." )
    endif ()
    message ( STATUS "---HSA_RUNTIME64_LIB (aqlprofile): ${HSA_RUNTIME64_LIB}" )
    set ( AQLPROFILE_WIN_HSA_LIB "${HSA_RUNTIME64_LIB}" )
  else ()
    # Locate the pre-built HIP import library (re-exports HSA runtime symbols)
    find_library ( AMDHIP64_LIB
      NAMES amdhip64
      HINTS "${AMDHIP64_LIB_DIR}"
    )
    if ( NOT AMDHIP64_LIB OR AMDHIP64_LIB STREQUAL "AMDHIP64_LIB-NOTFOUND" )
      message ( FATAL_ERROR "Failed to find amdhip64 import library for Windows build. Install HIP or set AMDHIP64_LIB_DIR to the directory containing amdhip64.lib." )
    endif ()
    message ( STATUS "---AMDHIP64_LIB: ${AMDHIP64_LIB}" )
    set ( AQLPROFILE_WIN_HSA_LIB "${AMDHIP64_LIB}" )
  endif ()
endif ()

target_link_libraries( ${TARGET_LIB}
  PRIVATE
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
    $<$<BOOL:${WIN32}>:${AQLPROFILE_WIN_HSA_LIB}>
  PUBLIC
    aqlprofile::headers
)

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
  add_subdirectory(src/pm4/tests)
endif()
