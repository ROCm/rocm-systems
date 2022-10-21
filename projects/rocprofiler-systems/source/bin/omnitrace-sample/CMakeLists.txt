# ------------------------------------------------------------------------------#
#
# omnitrace-sample target
#
# ------------------------------------------------------------------------------#

add_executable(omnitrace-sample ${CMAKE_CURRENT_LIST_DIR}/omnitrace-sample.cpp
                                ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(omnitrace-sample PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(omnitrace-sample PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    omnitrace-sample
    PRIVATE omnitrace::omnitrace-compile-definitions omnitrace::omnitrace-headers
            omnitrace::omnitrace-common-library)
set_target_properties(
    omnitrace-sample PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                                INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-sample)

install(
    TARGETS omnitrace-sample
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
