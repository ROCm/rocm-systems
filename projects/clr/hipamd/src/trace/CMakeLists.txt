# HIP API trace backend.
#
# Compiled into amdhip64 directly; there is no separate library. Only reached when
# HIP_TRACE_BACKEND is set to a backend that needs an implementation.

if(HIP_TRACE_BACKEND STREQUAL "tracelogging")
  target_sources(amdhip64 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace_etw.cpp)

  target_compile_definitions(amdhip64 PRIVATE HIP_TRACE_BACKEND_TRACELOGGING)

  # Trace headers live here; sources in this directory include them unqualified.
  target_include_directories(amdhip64 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # EventRegister/EventWriteTransfer, which the TraceLogging macros expand to.
  target_link_libraries(amdhip64 PRIVATE advapi32)
endif()
