set(TEST_SRC
    oob_module.cc)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.co
                COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                -x hip ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc
                -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                -o oob_kernel.co
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc)

set(ELF_FIXTURES
    ${CMAKE_CURRENT_BINARY_DIR}/elf_valid.co
    ${CMAKE_CURRENT_BINARY_DIR}/elf_huge_shnum.co
    ${CMAKE_CURRENT_BINARY_DIR}/elf_bad_shoff.co
    ${CMAKE_CURRENT_BINARY_DIR}/elf_table_spill.co
    ${CMAKE_CURRENT_BINARY_DIR}/elf_sh_overflow.co)

add_custom_command(
    OUTPUT  ${ELF_FIXTURES}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_evil_elfs.py
            ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.co
            ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.co
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_evil_elfs.py)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
    ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.co
    ${ELF_FIXTURES})

add_custom_target(gen_oob_kernel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/oob_kernel.co)
add_custom_target(gen_elf_fixtures DEPENDS ${ELF_FIXTURES})

hip_add_exe_to_target(NAME oob_module
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc)
add_dependencies(oob_module gen_oob_kernel gen_elf_fixtures)