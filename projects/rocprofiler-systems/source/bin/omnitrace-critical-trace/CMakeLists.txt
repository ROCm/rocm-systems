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

target_include_directories(omnitrace-critical-trace
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_definitions(omnitrace-critical-trace PRIVATE OMNITRACE_EXTERN_COMPONENTS=0)
target_link_libraries(omnitrace-critical-trace
                      PRIVATE omnitrace::omnitrace-interface-library)
set_target_properties(omnitrace-critical-trace PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(
    TARGETS omnitrace-critical-trace
    DESTINATION bin
    OPTIONAL)
