# ------------------------------------------------------------------------------#
#
# omnitrace dl library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)

add_library(omnitrace-dl-library SHARED)
add_library(omnitrace::omnitrace-dl-library ALIAS omnitrace-dl-library)

target_sources(omnitrace-dl-library PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dl.cpp)
target_link_libraries(omnitrace-dl-library PRIVATE ${dl_LIBRARY}
                                                   omnitrace::common-library)
target_include_directories(omnitrace-dl-library
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../omnitrace-user)

check_cxx_compiler_flag("-fno-exceptions" omnitrace_dl_library_fno_exceptions)
check_cxx_compiler_flag("-ftls-model=local-dynamic"
                        omnitrace_dl_library_ftls_module_local_dynamic)

if(OMNITRACE_BUILD_DEVELOPER)
    if(omnitrace_dl_library_fno_exceptions)
        target_compile_options(omnitrace-dl-library PRIVATE -fno-exceptions)
    endif()

    if(omnitrace_dl_library_ftls_module_local_dynamic)
        target_compile_options(omnitrace-dl-library PRIVATE -ftls-model=local-dynamic)
    endif()
endif()

omnitrace_target_compile_definitions(
    omnitrace-dl-library PRIVATE OMNITRACE_USE_OMPT=$<BOOL:${OMNITRACE_USE_OMPT}>)

set_target_properties(
    omnitrace-dl-library
    PROPERTIES OUTPUT_NAME omnitrace-dl
               CXX_VISIBILITY_PRESET "protected"
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

install(
    TARGETS omnitrace-dl-library
    EXPORT omnitrace-dl-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

install(
    EXPORT omnitrace-dl-library-targets
    FILE omnitrace-dl-library-targets.cmake
    NAMESPACE omnitrace::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace)
