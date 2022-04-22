# ------------------------------------------------------------------------------#
#
# omnitrace-avail target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-avail
    ${CMAKE_CURRENT_LIST_DIR}/avail.cpp ${CMAKE_CURRENT_LIST_DIR}/avail.hpp
    $<TARGET_OBJECTS:omnitrace::omnitrace-object-library>)

target_include_directories(omnitrace-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_definitions(omnitrace-avail PRIVATE OMNITRACE_EXTERN_COMPONENTS=0)
target_link_libraries(omnitrace-avail PRIVATE omnitrace::omnitrace-compile-definitions
                                              omnitrace::omnitrace-interface-library)
set_target_properties(
    omnitrace-avail
    PROPERTIES BUILD_RPATH "\$ORIGIN:${PROJECT_BINARY_DIR}:${CMAKE_BINARY_DIR}"
               INSTALL_RPATH_USE_LINK_PATH ON
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}")

install(
    TARGETS omnitrace-avail
    DESTINATION bin
    OPTIONAL)
