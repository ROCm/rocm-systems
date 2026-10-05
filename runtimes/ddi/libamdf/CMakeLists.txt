project(ROCmRuntimeAMDF VERSION 0.1 LANGUAGES C)

runtime_rust_library(runtime_amdf_shared
  PACKAGE libamdf LIBRARY amdf TYPE SHARED OUTPUT_NAME amdf
  VERSION "${PROJECT_VERSION}" SOVERSION "${PROJECT_VERSION_MAJOR}"
  SONAME_ENV ROCM_RUNTIME_AMDF_SONAME
)
runtime_rust_library(runtime_amdf_static
  PACKAGE libamdf LIBRARY amdf TYPE STATIC OUTPUT_NAME amdf
)
add_library(rocm_runtime::amdf_shared ALIAS runtime_amdf_shared)
add_library(rocm_runtime::amdf_static ALIAS runtime_amdf_static)
target_link_libraries(
    runtime_amdf_shared
    INTERFACE rocm_runtime::runtime_headers
)
target_link_libraries(
    runtime_amdf_static
    INTERFACE rocm_runtime::runtime_headers
)

if(BUILD_TESTING)
    foreach(kind shared static)
        add_executable(amdf_smoke_${kind} tests/abi/native_c_smoke.c)
        target_compile_features(amdf_smoke_${kind} PRIVATE c_std_11)
        target_link_libraries(
            amdf_smoke_${kind}
            PRIVATE rocm_runtime::amdf_${kind}
        )
        add_test(NAME amdf.smoke.${kind} COMMAND amdf_smoke_${kind})
        set_tests_properties(amdf.smoke.${kind} PROPERTIES LABELS "host;abi")
    endforeach()
    add_executable(amdf_layout tests/abi/amdf_layout.c)
    target_compile_features(amdf_layout PRIVATE c_std_11)
    target_link_libraries(amdf_layout PRIVATE rocm_runtime::runtime_headers)
    add_test(NAME amdf.layout COMMAND amdf_layout)
    set_tests_properties(amdf.layout PROPERTIES LABELS "host;abi")
endif()
