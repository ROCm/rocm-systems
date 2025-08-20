add_library (IcdLog icd_test_log.c)
if (WIN32)
    target_compile_definitions (IcdLog PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
