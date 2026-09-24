#
#
#
rocprofiler_activate_clang_tidy()

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
add_subdirectory(common)
add_subdirectory(rocprofiler-sdk)

if(WIN32)
    # aql_profile_v2.h is included transitively on Windows, so the AQLProfile directory is
    # still added: it generates aqlprofile/version.h and contributes the entry-point
    # stubs.
    add_subdirectory(aqlprofile)
    target_include_directories(rocprofiler-sdk-object-library
                               PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_subdirectory(output)

if(NOT WIN32)
    add_subdirectory(aqlprofile)
    add_subdirectory(rocprofiler-sdk-attach)

    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "roctx")
    add_subdirectory(rocprofiler-sdk-roctx)

    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "rocpd")
    add_subdirectory(rocprofiler-sdk-rocpd)

    # rocattach is only supported on x86_64 for now
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "rocattach")
        add_subdirectory(rocprofiler-sdk-rocattach)
    endif()

    add_subdirectory(python)
endif()

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")

# att-tool is the ATT decoder, which has no Windows build; rocprofiler-sdk-tool only needs
# the CodeobjLoadInfo header from it.
if(NOT WIN32)
    add_subdirectory(att-tool)
endif()

add_subdirectory(rocprofiler-sdk-tool)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
