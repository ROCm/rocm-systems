set(TEST_SRC
    loadlib_rtc.cc
    loadlib_co.cc
    library_negative.cc
)

add_custom_target(library_code_load.code
    COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/library_code_load.cc
            -o ${CMAKE_CURRENT_BINARY_DIR}/../library/library_code_load.code ${OFFLOAD_ARCH_STR}
            -I${HIP_PATH}/include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            --rocm-path=${ROCM_PATH})
set_property(GLOBAL APPEND PROPERTY
             G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/library_code_load.code)

if(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc)
else()
    hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS nvrtc)
endif()

add_dependencies(LibraryTests library_code_load.code)
