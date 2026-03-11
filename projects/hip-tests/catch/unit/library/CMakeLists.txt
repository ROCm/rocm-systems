set(TEST_SRC
    loadlib_rtc.cc
    #loadlib_co.cc TODO
    library_negative.cc
    hipKernelGetAttribute.cc
    lib_attribute.cc
)

if (HIP_PLATFORM MATCHES "amd")
  add_custom_target(library_code_load.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/library_code_load.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/library_code_load.code
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT})
add_custom_target(reverse_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/reverse_kernel.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/reverse_kernel.code
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT})
else()
  add_custom_target(library_code_load.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin -x cu ${CMAKE_CURRENT_SOURCE_DIR}/library_code_load.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/library_code_load.code ${OFFLOAD_ARCH_LIST}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${HIP_INCLUDE_DIR})
add_custom_target(reverse_kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} --fatbin -x cu ${CMAKE_CURRENT_SOURCE_DIR}/reverse_kernel.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/reverse_kernel.code ${OFFLOAD_ARCH_LIST}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${HIP_INCLUDE_DIR} )
endif()
set_property(GLOBAL APPEND PROPERTY
             G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/library_code_load.code)

set_property(GLOBAL APPEND PROPERTY
             G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/reverse_kernel.code)

if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc::hiprtc)
else()
  hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS nvrtc)
endif()

add_dependencies(LibraryTests library_code_load.code)
add_dependencies(LibraryTests reverse_kernel.code)
