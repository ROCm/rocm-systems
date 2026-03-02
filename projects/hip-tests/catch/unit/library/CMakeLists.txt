set(TEST_SRC
    loadlib_rtc.cc
    #loadlib_co.cc TODO
    library_negative.cc
)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _HIP_BUILD_TYPE_UPPER)
separate_arguments(HIP_DEVICE_BUILD_FLAGS NATIVE_COMMAND
  "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${_HIP_BUILD_TYPE_UPPER}}")

add_custom_target(library_code_load.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS} --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/library_code_load.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/library_code_load.code ${OFFLOAD_ARCH_LIST}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}) 
set_property(GLOBAL APPEND PROPERTY
             G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/library_code_load.code)

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
