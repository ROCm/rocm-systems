set(CONJUGATE_GRADIENT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/conjugate_gradient.cpp)
set_source_files_properties(${CONJUGATE_GRADIENT_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(conjugate_gradient_executable ${CONJUGATE_GRADIENT_SOURCES})
set_target_properties(
    conjugate_gradient_executable
    PROPERTIES OUTPUT_NAME conjugate_gradient
)

set(CG_KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/cg_kernels.hip)
set(CG_KERNEL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/cg_args.hpp)

set(CONJUGATE_GRADIENT_MODULES
    ${CMAKE_CURRENT_BINARY_DIR}/cg_module_a.hsaco
    ${CMAKE_CURRENT_BINARY_DIR}/cg_module_b.hsaco
)

set(CG_HSACO_ARCH_FLAGS)
foreach(HIP_ARCHITECTURE IN LISTS CMAKE_HIP_ARCHITECTURES)
    list(APPEND CG_HSACO_ARCH_FLAGS "--offload-arch=${HIP_ARCHITECTURE}")
endforeach()
list(REMOVE_DUPLICATES CG_HSACO_ARCH_FLAGS)

foreach(CG_MODULE_OUTPUT IN LISTS CONJUGATE_GRADIENT_MODULES)
    add_custom_command(
        OUTPUT ${CG_MODULE_OUTPUT}
        COMMAND
            ${CMAKE_HIP_COMPILER} -std=c++17 -g -O2 -x hip --offload-device-only
            ${CG_HSACO_ARCH_FLAGS} ${CG_KERNEL_SOURCE} -o ${CG_MODULE_OUTPUT}
        DEPENDS ${CG_KERNEL_SOURCE} ${CG_KERNEL_HEADER}
        COMMENT "Building ${CG_MODULE_OUTPUT}"
        VERBATIM
    )
endforeach()

add_custom_target(conjugate_gradient_modules DEPENDS ${CONJUGATE_GRADIENT_MODULES})

set(CONJUGATE_GRADIENT_MODULES ${CONJUGATE_GRADIENT_MODULES} PARENT_SCOPE)
