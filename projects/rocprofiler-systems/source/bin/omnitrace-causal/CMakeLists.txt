# ------------------------------------------------------------------------------#
#
# omnitrace-causal target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-causal
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-causal.hpp ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(omnitrace-causal PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(omnitrace-causal PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    omnitrace-causal
    PRIVATE omnitrace::omnitrace-compile-definitions omnitrace::omnitrace-headers
            omnitrace::omnitrace-common-library omnitrace::omnitrace-core)
set_target_properties(
    omnitrace-causal PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                                INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-causal)

install(
    TARGETS omnitrace-causal
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
