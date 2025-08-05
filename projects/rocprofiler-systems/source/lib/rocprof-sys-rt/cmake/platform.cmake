#
#
#
set(PLATFORM $ENV{PLATFORM})

set(VALID_PLATFORMS
    amd64-unknown-freebsd7.2
    i386-unknown-freebsd7.2
    i386-unknown-linux2.4
    ppc32_linux
    ppc64_linux
    x86_64-unknown-linux2.4
    aarch64-unknown-linux
)

if(NOT PLATFORM)
    set(INVALID_PLATFORM true)
else()
    list(FIND VALID_PLATFORMS ${PLATFORM} PLATFORM_FOUND)
    if(PLATFORM_FOUND EQUAL -1)
        set(INVALID_PLATFORM true)
    endif()
endif()

execute_process(
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/sysname
    OUTPUT_VARIABLE SYSNAME_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" "" SYSPLATFORM ${SYSNAME_OUT})

if(INVALID_PLATFORM)
    # Try to set it automatically
    execute_process(
        COMMAND ${CMAKE_CURRENT_LIST_DIR}/dynsysname ${SYSNAME_OUT}
        OUTPUT_VARIABLE DYNSYSNAME_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\n" "" PLATFORM ${DYNSYSNAME_OUT})
    rocprofiler_systems_message(
        STATUS "-- Attempting to automatically identify platform: ${PLATFORM}"
    )
endif()

list(FIND VALID_PLATFORMS ${PLATFORM} PLATFORM_FOUND)

if(PLATFORM_FOUND EQUAL -1)
    rocprofiler_systems_message(
        FATAL_ERROR
        "Error: unknown platform ${PLATFORM}; please set the PLATFORM environment variable to one of the following options: ${VALID_PLATFORMS}"
    )
endif()

if(PLATFORM MATCHES i386)
    set(ARCH_DEFINES arch_x86)
    list(
        APPEND
        CAP_DEFINES
        cap_fixpoint_gen
        cap_noaddr_gen
        cap_stripped_binaries
        cap_tramp_liveness
        cap_virtual_registers
        cap_stack_mods
    )
elseif(PLATFORM MATCHES x86_64 OR PLATFORM MATCHES amd64)
    set(ARCH_DEFINES arch_x86_64 arch_64bit)
    list(
        APPEND
        CAP_DEFINES
        cap_32_64
        cap_fixpoint_gen
        cap_noaddr_gen
        cap_registers
        cap_stripped_binaries
        cap_tramp_liveness
        cap_stack_mods
    )
elseif(PLATFORM MATCHES ppc32)
    set(ARCH_DEFINES arch_power)
    list(APPEND CAP_DEFINES cap_registers)
elseif(PLATFORM MATCHES ppc64)
    set(ARCH_DEFINES arch_power arch_64bit)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
    list(APPEND CAP_DEFINES cap_32_64 cap_registers cap_toc_64)

    if(SYSPLATFORM MATCHES ppc64le)
        list(APPEND CAP_DEFINES arch_ppc_little_endian)
    endif()
elseif(PLATFORM MATCHES aarch64)
    # set(ARCH_DEFINES aarch_64 arch_64bit)
    set(ARCH_DEFINES arch_aarch64 arch_64bit)
    list(APPEND CAP_DEFINES cap_32_64 cap_registers)
endif()

if(PLATFORM MATCHES linux)
    set(OS_DEFINES os_linux)
    list(
        APPEND
        CAP_DEFINES
        cap_async_events
        cap_binary_rewriter
        cap_dwarf
        cap_mutatee_traps
        cap_ptrace
    )
    set(BUG_DEFINES bug_syscall_changepc_rewind bug_force_terminate_failure)
elseif(PLATFORM MATCHES cnl)
    set(OS_DEFINES os_linux os_cnl)
    list(
        APPEND
        CAP_DEFINES
        cap_async_events
        cap_binary_rewriter
        cap_dwarf
        cap_mutatee_traps
        cap_ptrace
    )
    set(BUG_DEFINES bug_syscall_changepc_rewind)
elseif(PLATFORM MATCHES freebsd)
    set(OS_DEFINES os_freebsd)
    list(APPEND CAP_DEFINES cap_binary_rewriter cap_dwarf cap_mutatee_traps)
    set(BUG_DEFINES
        bug_freebsd_missing_sigstop
        bug_freebsd_mt_suspend
        bug_freebsd_change_pc
        bug_phdrs_first_page
        bug_syscall_changepc_rewind
    )
elseif(PLATFORM STREQUAL i386-unknown-nt4.0)
    set(OS_DEFINES os_windows)
    list(APPEND CAP_DEFINES cap_mem_emulation cap_mutatee_traps)
endif()

if(PLATFORM STREQUAL i386-unknown-linux2.4)
    set(OLD_DEFINES i386_unknown_linux2_0)
elseif(PLATFORM STREQUAL x86_64-unknown-linux2.4)
    set(OLD_DEFINES x86_64_unknown_linux2_4)
elseif(PLATFORM STREQUAL ppc32_linux)
    set(OLD_DEFINES ppc32_linux)
    set(BUG_DEFINES ${BUG_DEFINES} bug_registers_after_exit)
elseif(PLATFORM STREQUAL ppc64_linux)
    set(OLD_DEFINES ppc64_linux)
    set(BUG_DEFINES ${BUG_DEFINES} bug_registers_after_exit)
elseif(PLATFORM STREQUAL x86_64_cnl)
    set(OLD_DEFINES x86_64_cnl x86_64_unknown_linux2_4)
elseif(PLATFORM STREQUAL i386-unknown-freebsd7.2)
    set(OLD_DEFINES i386_unknown_freebsd7_0)
elseif(PLATFORM STREQUAL amd64-unknown-freebsd7.2)
    set(OLD_DEFINES amd64_unknown_freebsd7_0)
elseif(PLATFORM STREQUAL i386-unknown-nt4.0)
    set(OLD_DEFINES i386_unknown_nt4_0)
elseif(PLATFORM STREQUAL aarch64-unknown-linux)
    set(OLD_DEFINES aarch64_unknown_linux)
else(PLATFORM STREQUAL i386-unknown-linux2.4)
    dyninst_message(FATAL_ERROR "Unknown platform: ${PLATFORM}")
endif()

if(Thread_DB_FOUND)
    dyninst_message(STATUS "-- Enabling ThreadDB support")
    list(APPEND CAP_DEFINES cap_thread_db)
endif()

set(UNIFIED_DEFINES
    ${CAP_DEFINES}
    ${BUG_DEFINES}
    ${ARCH_DEFINES}
    ${OS_DEFINES}
    ${OLD_DEFINES}
)

list(REMOVE_DUPLICATES UNIFIED_DEFINES)
