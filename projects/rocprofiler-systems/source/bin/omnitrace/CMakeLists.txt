# ------------------------------------------------------------------------------#
#
# omnitrace-exe target
#
# ------------------------------------------------------------------------------#

add_executable(omnitrace-exe ${_EXCLUDE})

target_sources(
    omnitrace-exe
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/omnitrace.cpp
            ${CMAKE_CURRENT_LIST_DIR}/details.cpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.cpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.cpp
            ${CMAKE_CURRENT_LIST_DIR}/omnitrace.hpp
            ${CMAKE_CURRENT_LIST_DIR}/info.hpp
            ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
            ${CMAKE_CURRENT_LIST_DIR}/function_signature.hpp
            ${CMAKE_CURRENT_LIST_DIR}/module_function.hpp)

target_link_libraries(
    omnitrace-exe
    PRIVATE omnitrace::omnitrace-headers
            omnitrace::omnitrace-dyninst
            omnitrace::omnitrace-compile-options
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>)

set_target_properties(
    omnitrace-exe
    PROPERTIES
        OUTPUT_NAME omnitrace
        BUILD_RPATH "\$ORIGIN:${PROJECT_BINARY_DIR}:${CMAKE_BINARY_DIR}"
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/timemory/libunwind:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib"
    )

if(CMAKE_BUILD_TYPE MATCHES "^(DEBUG|Debug)")
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(omnitrace-exe PRIVATE ${_FLAGS})
endif()

install(
    TARGETS omnitrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
