# Kernel Module Build
# CMake wrapper around Kbuild for building the kernel module

set(MODULE_KO "amdgpu_pmu.ko")
set(KBUILD_CMD ${CMAKE_MAKE_PROGRAM})

# Get the current source directory for the kernel module
set(KBUILD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KBUILD_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

# List of source files and headers that need to be linked
set(KERNEL_MODULE_SOURCES
    pmu_main.c
    pmu_events.c
    amdgpu_pmu.h
    kfd_test.c
    kfd_test.h
    aql_perf.c
    aql_perf.h
    aql_packet_ops.c
    aql_error_recovery.c
    aql_pmu_integration.c
    example_kfd_usage.c
)

# Create symbolic links to source files in build directory
foreach(src_file ${KERNEL_MODULE_SOURCES})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${KBUILD_SRC_DIR}/${src_file}
            ${KBUILD_BIN_DIR}/${src_file}
    )
endforeach()

# Create symbolic link to aql_c directory
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${KBUILD_SRC_DIR}/aql_c
        ${KBUILD_BIN_DIR}/aql_c
)

# Create Kbuild file from the Makefile (extract only the kernel build parts)
file(WRITE ${KBUILD_BIN_DIR}/Kbuild
"# Kernel build configuration
obj-m := amdgpu_pmu.o

# Include AMD GPU PMU, KFD test, and minimal AQL integration
amdgpu_pmu-y := pmu_main.o pmu_events.o kfd_test.o \\
              aql_perf.o aql_packet_ops.o aql_error_recovery.o aql_pmu_integration.o \\
              aql_c/arch_creator.o aql_c/gfx12_creator.o \\
              aql_c/pm4_packets.o aql_c/counter_registry.o aql_c/gfx12_events.o \\
              aql_c/packet_generation.o

# Compiler flags
ccflags-y := -DDEBUG -g -Wall -D__KERNEL__ -I\$(src)/aql_c
"
)

# Custom command to build the kernel module
add_custom_command(
    OUTPUT ${KBUILD_BIN_DIR}/${MODULE_KO}
    COMMAND ${CMAKE_COMMAND} -E echo "Building kernel module..."
    COMMAND ${KBUILD_CMD} -C ${KERNEL_BUILD_DIR} M=${KBUILD_BIN_DIR} modules
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying module build..."
    COMMAND test -f ${KBUILD_BIN_DIR}/${MODULE_KO}
    COMMAND sh -c "strings ${KBUILD_BIN_DIR}/${MODULE_KO} | grep vermagic || echo 'WARNING: No vermagic found'"
    WORKING_DIRECTORY ${KBUILD_BIN_DIR}
    DEPENDS
        ${KBUILD_SRC_DIR}/pmu_main.c
        ${KBUILD_SRC_DIR}/pmu_events.c
        ${KBUILD_SRC_DIR}/kfd_test.c
        ${KBUILD_SRC_DIR}/aql_perf.c
        ${KBUILD_SRC_DIR}/aql_packet_ops.c
        ${KBUILD_SRC_DIR}/aql_error_recovery.c
        ${KBUILD_SRC_DIR}/aql_pmu_integration.c
        ${KBUILD_SRC_DIR}/aql_c/arch_creator.c
        ${KBUILD_SRC_DIR}/aql_c/gfx12_creator.c
        ${KBUILD_SRC_DIR}/aql_c/pm4_packets.c
        ${KBUILD_SRC_DIR}/aql_c/counter_registry.c
        ${KBUILD_SRC_DIR}/aql_c/gfx12_events.c
        ${KBUILD_SRC_DIR}/aql_c/packet_generation.c
        ${KBUILD_SRC_DIR}/Makefile
    COMMENT "Building kernel module ${MODULE_KO}"
    VERBATIM
)

# Custom target for the kernel module
add_custom_target(kernel_module ALL
    DEPENDS ${KBUILD_BIN_DIR}/${MODULE_KO}
)

# Clean target
add_custom_target(kernel_module_clean
    COMMAND ${KBUILD_CMD} -C ${KERNEL_BUILD_DIR} M=${KBUILD_BIN_DIR} clean
    WORKING_DIRECTORY ${KBUILD_BIN_DIR}
    COMMENT "Cleaning kernel module build artifacts"
)

# Install target
install(
    FILES ${KBUILD_BIN_DIR}/${MODULE_KO}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/modules/${KERNEL_VERSION}/extra
    OPTIONAL
)

# Add post-install script to run depmod
install(CODE "execute_process(COMMAND depmod -a)")

# Module load/unload helper targets
add_custom_target(load-module
    COMMAND sudo insmod ${KBUILD_BIN_DIR}/${MODULE_KO}
    DEPENDS kernel_module
    COMMENT "Loading kernel module"
)

add_custom_target(unload-module
    COMMAND sudo rmmod amdgpu_pmu || true
    COMMENT "Unloading kernel module"
)

add_custom_target(reload-module
    DEPENDS unload-module kernel_module
    COMMAND sudo insmod ${KBUILD_BIN_DIR}/${MODULE_KO}
    COMMENT "Reloading kernel module"
)

# Module info target
add_custom_target(module-info
    COMMAND ${CMAKE_COMMAND} -E echo "Module: ${MODULE_NAME}"
    COMMAND ${CMAKE_COMMAND} -E echo "Version: ${MODULE_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo "Kernel: ${KERNEL_VERSION}"
    COMMAND lsmod | grep amdgpu_pmu || echo "Module not loaded"
    COMMENT "Displaying module information"
)
