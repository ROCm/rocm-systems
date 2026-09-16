set(TEST_SRC
    oob_module.cc)

# Only the valid code object is compiled and installed. The malformed variants
# that exercise getElfSize's rejection paths are synthesized in-process by the
# test (see oob_module.cc); they are deliberately NOT generated or installed as
# fixtures, because installing corrupt ELFs breaks DEB packaging (dh_makeshlibs
# runs `objdump -p` on every installed ELF and aborts on a bad header).
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.code
                COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc
                -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                -o ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.code
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
    ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.code)

add_custom_target(gen_oob_kernel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.code)

# Valid compressed bundle used by oob_module.cc's false-positive guard.
# Built from a local copyKernel.cc (self-contained, no reach into unit/module).
# All fixtures install flat into one directory (see packaging/CMakeLists.txt), so
# the oob-prefixed name avoids colliding with module's copyKernelCompressed.code.
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/oob_copyKernelCompressed.code
                COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} -mcode-object-version=5 --offload-compress --cuda-device-only
                -x hip ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                -o ${CMAKE_CURRENT_BINARY_DIR}/oob_copyKernelCompressed.code
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
    ${CMAKE_CURRENT_BINARY_DIR}/oob_copyKernelCompressed.code)

add_custom_target(oob_copyKernelCompressed_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/oob_copyKernelCompressed.code)

hip_add_exe_to_target(NAME oob_module
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc)
add_dependencies(oob_module gen_oob_kernel oob_copyKernelCompressed_code)
