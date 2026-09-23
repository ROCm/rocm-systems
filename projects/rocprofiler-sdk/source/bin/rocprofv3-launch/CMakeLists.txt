#
# rocprofv3-launch: runs the ETW consumer beside the application instead of inside it
#
rocprofiler_activate_clang_tidy()

add_executable(rocprofv3-launch)
add_executable(rocprofiler-sdk::rocprofv3-launch ALIAS rocprofv3-launch)

target_sources(rocprofv3-launch PRIVATE main.cpp)

target_link_libraries(
    rocprofv3-launch
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck)

set_target_properties(
    rocprofv3-launch PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

install(
    TARGETS rocprofv3-launch
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
