# ------------------------------------------------------------------------------#
#
# omnitrace-run target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-run
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-run.cpp
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-run.hpp ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(omnitrace-run PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(omnitrace-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    omnitrace-run
    PRIVATE omnitrace::omnitrace-compile-definitions omnitrace::omnitrace-headers
            omnitrace::omnitrace-common-library omnitrace::omnitrace-core
            omnitrace::omnitrace-sanitizer)
set_target_properties(
    omnitrace-run PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                             INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-run)

install(
    TARGETS omnitrace-run
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
