add_library(cltrace SHARED cltrace.cpp)

if(WIN32)
  target_sources(cltrace PRIVATE cltrace.def)
else()
  target_link_libraries(cltrace PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/cltrace.map")
  set_target_properties(cltrace PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cltrace.map")
endif()

target_compile_definitions(cltrace PRIVATE CL_TARGET_OPENCL_VERSION=220)

target_include_directories(cltrace PRIVATE ${CMAKE_SOURCE_DIR}/opencl ${OPENCL_ICD_LOADER_HEADERS_DIR} ${ROCCLR_INCLUDE_DIR})

INSTALL(TARGETS cltrace
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
