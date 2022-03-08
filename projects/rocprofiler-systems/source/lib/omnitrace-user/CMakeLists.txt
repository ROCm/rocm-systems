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
    omnitrace-user-library PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/user.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/user.h)
target_include_directories(
    omnitrace-user-library PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

check_cxx_compiler_flag("-fno-exceptions" omnitrace_user_library_fno_exceptions)
check_cxx_compiler_flag("-ftls-model=local-dynamic"
                        omnitrace_user_library_ftls_module_local_dynamic)

if(OMNITRACE_BUILD_DEVELOPER)
    if(omnitrace_user_library_fno_exceptions)
        target_compile_options(omnitrace-user-library PRIVATE -fno-exceptions)
    endif()

    if(omnitrace_user_library_ftls_module_local_dynamic)
        target_compile_options(omnitrace-user-library PRIVATE -ftls-model=local-dynamic)
    endif()
endif()

set_target_properties(
    omnitrace-user-library
    PROPERTIES OUTPUT_NAME omnitrace-user
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

install(
    TARGETS omnitrace-user-library
    EXPORT omnitrace-user-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/user.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnitrace
    OPTIONAL)

install(
    EXPORT omnitrace-user-library-targets
    FILE omnitrace-user-library-targets.cmake
    NAMESPACE omnitrace::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace)
