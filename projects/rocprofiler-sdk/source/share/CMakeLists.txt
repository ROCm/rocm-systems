#
#
#

add_subdirectory(rocprofiler-sdk)
add_subdirectory(rocprofiler-sdk-rocpd)

install(
    FILES bash-completion/completions/rocprofv3
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/bash-completion/completions
    COMPONENT tools)
