# ------------------------------------------------------------------------------#
#
# omnitrace-critical-trace target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-critical-trace
    ${CMAKE_CURRENT_LIST_DIR}/critical-trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/critical-trace.hpp
    $<TARGET_OBJECTS:omnitrace::omnitrace-object-library>)

target_include_directories(omnitrace-critical-trace PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(omnitrace-critical-trace PRIVATE OMNITRACE_EXTERN_COMPONENTS=0)
target_link_libraries(
    omnitrace-critical-trace
    PRIVATE omnitrace::omnitrace-compile-definitions
            omnitrace::omnitrace-interface-library omnitrace::omnitrace-headers
            omnitrace::omnitrace-timemory)
set_target_properties(
    omnitrace-critical-trace
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}")

install(
    TARGETS omnitrace-critical-trace
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
