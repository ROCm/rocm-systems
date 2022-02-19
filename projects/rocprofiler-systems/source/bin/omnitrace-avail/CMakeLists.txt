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
target_link_libraries(omnitrace-avail PRIVATE omnitrace::omnitrace-interface-library)
set_target_properties(omnitrace-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(
    TARGETS omnitrace-avail
    DESTINATION bin
    OPTIONAL)
