# ------------------------------------------------------------------------------#
#
# omnitrace object library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-object-library OBJECT)
add_library(omnitrace::omnitrace-object-library ALIAS omnitrace-object-library)

target_sources(
    omnitrace-object-library
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/library.cpp ${CMAKE_CURRENT_LIST_DIR}/regions.cpp
            ${CMAKE_CURRENT_LIST_DIR}/progress.cpp ${CMAKE_CURRENT_LIST_DIR}/api.cpp
            ${CMAKE_CURRENT_LIST_DIR}/api.hpp)

add_subdirectory(library)

target_link_libraries(omnitrace-object-library
                      PRIVATE omnitrace::omnitrace-interface-library)

if(OMNITRACE_DYNINST_API_RT)
    get_filename_component(OMNITRACE_DYNINST_API_RT_DIR "${OMNITRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

# ------------------------------------------------------------------------------#
#
# omnitrace static library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-static-library STATIC $<TARGET_OBJECTS:omnitrace-object-library>)
add_library(omnitrace::libomnitrace-static ALIAS omnitrace-static-library)

target_link_libraries(
    omnitrace-static-library
    PRIVATE omnitrace::omnitrace-interface-library omnitrace::omnitrace-core
            omnitrace::omnitrace-binary)

set_target_properties(omnitrace-static-library PROPERTIES OUTPUT_NAME omnitrace)

# ------------------------------------------------------------------------------#
#
# omnitrace shared library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-shared-library SHARED $<TARGET_OBJECTS:omnitrace-object-library>)
add_library(omnitrace::libomnitrace-shared ALIAS omnitrace-shared-library)
add_library(omnitrace::omnitrace-library ALIAS omnitrace-shared-library)

target_link_libraries(
    omnitrace-shared-library
    PRIVATE omnitrace::omnitrace-interface-library omnitrace::omnitrace-core
            omnitrace::omnitrace-binary)

set_target_properties(
    omnitrace-shared-library
    PROPERTIES OUTPUT_NAME omnitrace
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               INSTALL_RPATH "${OMNITRACE_LIB_INSTALL_RPATH}")

omnitrace_strip_target(omnitrace-shared-library)

install(
    TARGETS omnitrace-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
