# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipCreateTextureObject_ArgValidation.cc
    hipCreateTextureObject_Linear.cc
    hipCreateTextureObject_Pitch2D.cc
    hipCreateTextureObject_Array.cc
    hipTextureObjFetchVector.cc
    hipNormalizedFloatValueTex.cc
    hipTextureObj2D.cc
    hipSimpleTexture3D.cc
    hipTextureRef2D.cc
    hipSimpleTexture1DLayered.cc
    hipSimpleTexture2DLayered.cc
    hipBindTexture.cc
    hipBindTexture2D.cc
    hipTex1DFetchCheckModes.cc
    hipGetChanDesc.cc
    hipGetTextureAlignmentOffset.cc
    hipGetTextureReference.cc
    hipTexObjPitch.cc
    hipTexRefSetArray.cc
    hipTexRefGetArray.cc
    hipTexRefGetBorderColor.cc
    hipTexRefSetBorderColor.cc
    hipTexRefGetAddress.cc
    hipTexRefSetAddress.cc
    hipTexRefGetFormat.cc
    hipTexRefSetFormat.cc
    hipTextureObj1DFetch.cc
    hipTextureObj1DCheckModes.cc
    hipTextureObj2DCheckModes.cc
    hipTextureObj3DCheckModes.cc
    hipTextureObj1DCheckSRGBModes.cc
    hipTextureObj2DCheckSRGBModes.cc
    hipTexObjectCreate.cc
    hipTexObjectTests.cc
    hipTextureObjectTests.cc
    hipTexRefSetAddress2D.cc
    hipTexRefSetMaxAnisotropy.cc
    hipTexRefGetMaxAnisotropy.cc
    hipUnbindTexture.cc
    hipTexRefSetFlags.cc
    hipBindTextureToArray.cc
    hipTexRefGetFlags.cc
    hipTexRefSetAddressMode.cc
    hipTexRefGetAddressMode.cc
    hipTexRefSetGetFilterMode.cc
    hipDeviceGetTexture1DLinearMaxWidth.cc
    hipMipmappedArrayGetMemoryRequirements.cc
)

# tests not for gfx90a+
set(NOT_FOR_gfx90a_AND_ABOVE_TEST
    tex1D.cc
    tex1DGrad.cc
    tex1DLayeredGrad.cc
    tex1Dfetch.cc
    tex1DLayered.cc
    tex2D.cc
    tex2DGrad.cc
    tex2DLayeredGrad.cc
    tex2Dgather.cc
    tex2DLayered.cc
    tex3D.cc
    tex3DGrad.cc
    hipBindTextureToMipmappedArray.cc
    hipMallocMipmappedArray.cc
    hipFreeMipmappedArray.cc
    hipGetMipmappedArrayLevel.cc
    hipMipmappedArrayCreate.cc
    hipMipmappedArrayDestroy.cc
    hipMipmappedArrayGetLevel.cc
    tex1DLod.cc
    tex1DLayeredLod.cc
    tex2DLod.cc
    tex2DLayeredLod.cc
    tex3DLod.cc
    hipTexRefSetGetMipmapFilterMode.cc
    hipTexRefSetGetMipmapLevelBias.cc
    hipTexRefSetGetMipmapLevelClamp.cc
    hipTexRefSetGetMipmappedArray.cc
)

if (HIP_PLATFORM MATCHES "amd")
  set(${TEST_SRC} ${TEST_SRC} hipTextureMipmapObj1D.cc
      hipTextureMipmapObj2D.cc
    hipTextureMipmapObj3D.cc
)
endif()

set(gfx90a_AND_ABOVE_TARGETS gfx90a gfx942 gfx950)
function(CheckRejectedArchs OFFLOAD_ARCH_STR_LOCAL)
  set(ARCH_CHECK -1 PARENT_SCOPE)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
    string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
    if (CMAKE_MATCH_COUNT EQUAL 1)
      if (CMAKE_MATCH_1 IN_LIST gfx90a_AND_ABOVE_TARGETS)
        set(ARCH_CHECK 1 PARENT_SCOPE)
      endif()         # CMAKE_MATCH_1
    endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
endfunction()  # CheckAcceptedArchs

if (HIP_PLATFORM MATCHES "amd")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tex_ref_get_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only -Wno-deprecated-declarations
                   -x hip ${CMAKE_CURRENT_SOURCE_DIR}/tex_ref_get_module.cc
                   -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                   -o tex_ref_get_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tex_ref_get_module.cc)
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tex_ref_get_module.code
                   COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                   -x cu ${CMAKE_CURRENT_SOURCE_DIR}/tex_ref_get_module.cc
                   -I${HIP_INCLUDE_DIR}
                   -o tex_ref_get_module.code
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tex_ref_get_module.cc)
endif()
add_custom_target(tex_ref_get_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tex_ref_get_module.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/tex_ref_get_module.code)

if(HIP_PLATFORM MATCHES "amd")
  if (DEFINED OFFLOAD_ARCH_STR)
    CheckRejectedArchs(${OFFLOAD_ARCH_STR})
  elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
    CheckRejectedArchs($ENV{HCC_AMDGPU_TARGET})
  else()
    set(ARCH_CHECK -1)
  endif()
  if(${ARCH_CHECK} EQUAL -1)
    message(STATUS "Adding test: ${NOT_FOR_gfx90a_AND_ABOVE_TEST}")
    set(TEST_SRC ${TEST_SRC} ${NOT_FOR_gfx90a_AND_ABOVE_TEST})
  else()
    message(STATUS "Removing test: ${NOT_FOR_gfx90a_AND_ABOVE_TEST}")
  endif()
else()
  set(TEST_SRC ${TEST_SRC} ${NOT_FOR_gfx90a_AND_ABOVE_TEST})
endif()

get_filename_component(HIP_TEST_LABELS ${CMAKE_CURRENT_SOURCE_DIR} NAME)
hip_add_exe_to_target(NAME TextureTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

target_compile_options(TextureTest PRIVATE -Wno-deprecated-declarations)
add_dependencies(TextureTest tex_ref_get_module)
