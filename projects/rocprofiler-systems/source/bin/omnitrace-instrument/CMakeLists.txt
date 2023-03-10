# ------------------------------------------------------------------------------#
#
# omnitrace-instrument target (formerly omnitrace-exe target prior to 1.8.1)
#
# ------------------------------------------------------------------------------#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(omnitrace-instrument)

target_sources(
    omnitrace-instrument
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/details.cpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.cpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.hpp
            ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
            ${CMAKE_CURRENT_LIST_DIR}/info.hpp
            ${CMAKE_CURRENT_LIST_DIR}/internal_libs.cpp
            ${CMAKE_CURRENT_LIST_DIR}/internal_libs.hpp
            ${CMAKE_CURRENT_LIST_DIR}/log.cpp
            ${CMAKE_CURRENT_LIST_DIR}/log.hpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.cpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.hpp
            ${CMAKE_CURRENT_LIST_DIR}/omnitrace-instrument.cpp
            ${CMAKE_CURRENT_LIST_DIR}/omnitrace-instrument.hpp)

target_link_libraries(
    omnitrace-instrument
    PRIVATE omnitrace::omnitrace-headers
            omnitrace::omnitrace-dyninst
            omnitrace::omnitrace-compile-options
            omnitrace::omnitrace-compile-definitions
            omnitrace::omnitrace-sanitizer
            timemory::timemory-headers
            timemory::timemory-extensions
            timemory::timemory-core)

set_target_properties(
    omnitrace-instrument
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH ON)

if(OMNITRACE_BUILD_DYNINST)
    target_compile_definitions(omnitrace-instrument PRIVATE OMNITRACE_BUILD_DYNINST=1)
endif()

omnitrace_strip_target(omnitrace-instrument)

if(CMAKE_BUILD_TYPE MATCHES "^(DEBUG|Debug)")
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(omnitrace-instrument PRIVATE ${_FLAGS})
endif()

install(
    TARGETS omnitrace-instrument
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
