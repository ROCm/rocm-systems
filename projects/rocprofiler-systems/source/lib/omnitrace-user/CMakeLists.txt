# ------------------------------------------------------------------------------#
#
# omnitrace user library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(omnitrace-user-library SHARED)
add_library(omnitrace::omnitrace-user-library ALIAS omnitrace-user-library)

target_sources(
    omnitrace-user-library
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/user.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/user.h
            ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/types.h)
target_include_directories(
    omnitrace-user-library PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(omnitrace-user-library
                      PRIVATE $<BUILD_INTERFACE:omnitrace::omnitrace-compile-definitions>)

set_target_properties(
    omnitrace-user-library
    PROPERTIES OUTPUT_NAME omnitrace-user
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

omnitrace_strip_target(omnitrace-user-library)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/user.h
              ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/types.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnitrace)

install(
    TARGETS omnitrace-user-library
    EXPORT omnitrace-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
