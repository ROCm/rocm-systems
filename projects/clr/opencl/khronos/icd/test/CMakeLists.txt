include_directories (./inc)

add_subdirectory (log)
add_subdirectory (driver_stub)
add_subdirectory (loader_test)

add_test (NAME opencl_icd_loader_test COMMAND icd_loader_test)

if (WIN32)
    add_subdirectory (local_path_test)
    add_test (NAME opencl_icd_local_path_test COMMAND icd_local_path_test)
endif ()
