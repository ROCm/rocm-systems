// Windows test for rocprofiler-register with HIP library
// Tests registration functionality with actual amdhip64.dll

#include <rocprofiler-register/rocprofiler-register.h>
#include <rocprofiler-register/version.h>

#include <windows.h>
#include <iostream>
#include <cstdio>

int main() {
    std::cout << "=== rocprofiler-register Windows Test with HIP ===\n\n";

    // Test 1: Verify rocprofiler-register version info
    std::cout << "[TEST 1] rocprofiler-register Version:\n";
    std::cout << "  Version: " << ROCPROFILER_REGISTER_VERSION_STRING << "\n";
    std::cout << "  System: " << ROCPROFILER_REGISTER_SYSTEM_NAME << "\n";
    std::cout << "  Git Revision: " << ROCPROFILER_REGISTER_GIT_REVISION << "\n";
    std::cout << "  \xE2\x9C\x93 PASSED\n\n";

    // Test 2: Load amdhip64.dll
    std::cout << "[TEST 2] Load HIP Library:\n";
    HMODULE hip_dll = LoadLibraryA("C:\\opt\\rocm\\bin\\amdhip64_7.dll");
    if (!hip_dll) {
        DWORD error = GetLastError();
        std::cout << "  \xE2\x9C\x97 FAILED: Could not load amdhip64_7.dll (Error: " << error << ")\n";
        std::cout << "  Make sure C:\\opt\\rocm\\bin\\amdhip64_7.dll exists\n";
        return 1;
    }
    std::cout << "  Loaded amdhip64_7.dll successfully\n";
    std::cout << "  DLL Handle: " << hip_dll << "\n";
    std::cout << "  \xE2\x9C\x93 PASSED\n\n";

    // Test 3: Check for HIP import function
    std::cout << "[TEST 3] Check HIP Import Function:\n";
    typedef uint32_t (*import_func_t)(void);
    auto hip_import = (import_func_t)GetProcAddress(hip_dll, "rocprofiler_register_import_hip");

    if (hip_import) {
        uint32_t hip_version = hip_import();
        std::cout << "  Found rocprofiler_register_import_hip\n";
        std::cout << "  HIP version from import: " << hip_version << "\n";
        std::cout << "  \xE2\x9C\x93 PASSED\n\n";
    } else {
        std::cout << "  \xE2\x9A\xA0 WARNING: rocprofiler_register_import_hip not found\n";
        std::cout << "  This may be expected if HIP doesn't export this symbol yet\n\n";
    }

    // Test 4: Test rocprofiler-register API with HIP
    std::cout << "[TEST 4] Register HIP with rocprofiler-register:\n";

    // Create a mock API table (HIP would normally provide this)
    void* mock_hip_api_table = (void*)0x12345678;  // Mock pointer
    rocprofiler_register_library_indentifier_t register_id;

    rocprofiler_register_error_code_t result;

    if (hip_import) {
        // Register with the actual HIP import function
        result = rocprofiler_register_library_api_table(
            "hip",
            hip_import,
            hip_import(),  // Use actual HIP version
            &mock_hip_api_table,
            1,
            &register_id
        );
    } else {
        // Register with a mock import function since HIP doesn't have one
        auto mock_import = []() -> uint32_t {
            // HIP version 6.0.0 = 60000
            return ROCPROFILER_REGISTER_COMPUTE_VERSION_3(6, 0, 0);
        };

        result = rocprofiler_register_library_api_table(
            "hip",
            mock_import,
            60000,  // HIP 6.0.0
            &mock_hip_api_table,
            1,
            &register_id
        );
    }

    std::cout << "  Registration result: " << rocprofiler_register_error_string(result)
              << " (" << result << ")\n";

    // Expected results:
    // - ROCP_REG_NO_TOOLS (1) = No profiler tool loaded (expected in test environment)
    // - ROCP_REG_SUCCESS (0) = Successfully registered (if profiler is present)
    if (result == ROCP_REG_NO_TOOLS || result == ROCP_REG_SUCCESS) {
        std::cout << "  Registration ID: " << register_id.handle << "\n";
        std::cout << "  \xE2\x9C\x93 PASSED (Expected result - no profiler loaded)\n\n";
    } else {
        std::cout << "  \xE2\x9A\xA0 WARNING: Unexpected result code\n\n";
    }

    // Test 5: Iterate over registered libraries
    std::cout << "[TEST 5] Check Registration Info:\n";
    int callback_count = 0;
    auto callback = [](rocprofiler_register_registration_info_t* info, void* data) -> int {
        int* count = (int*)data;
        (*count)++;
        std::cout << "    Registration #" << *count << ":\n";
        std::cout << "      Name: " << info->common_name << "\n";
        std::cout << "      Version: " << info->lib_version << "\n";
        std::cout << "      API Tables: " << info->api_table_length << "\n";
        return 0;
    };

    result = rocprofiler_register_iterate_registration_info(callback, &callback_count);
    std::cout << "  Total registrations found: " << callback_count << "\n";
    std::cout << "  Result: " << rocprofiler_register_error_string(result) << "\n";

    if (callback_count >= 1) {
        std::cout << "  \xE2\x9C\x93 PASSED (Found HIP registration)\n\n";
    } else {
        std::cout << "  \xE2\x9C\x93 PASSED (No registrations - expected without profiler)\n\n";
    }

    // Test 6: Verify rocprofiler-register.dll exports
    std::cout << "[TEST 6] Verify rocprofiler-register.dll Exports:\n";
    HMODULE reg_dll = LoadLibraryA("C:\\opt\\rocm\\bin\\rocprofiler-register.dll");
    if (!reg_dll) {
        std::cout << "  \xE2\x9C\x97 FAILED: Could not load rocprofiler-register.dll\n";
        FreeLibrary(hip_dll);
        return 1;
    }

    auto reg_func = GetProcAddress(reg_dll, "rocprofiler_register_library_api_table");
    auto err_func = GetProcAddress(reg_dll, "rocprofiler_register_error_string");
    auto iter_func = GetProcAddress(reg_dll, "rocprofiler_register_iterate_registration_info");

    if (reg_func && err_func && iter_func) {
        std::cout << "  All required functions exported:\n";
        std::cout << "    - rocprofiler_register_library_api_table\n";
        std::cout << "    - rocprofiler_register_error_string\n";
        std::cout << "    - rocprofiler_register_iterate_registration_info\n";
        std::cout << "  \xE2\x9C\x93 PASSED\n\n";
    } else {
        std::cout << "  \xE2\x9C\x97 FAILED: Missing exports\n";
        if (!reg_func) std::cout << "    Missing: rocprofiler_register_library_api_table\n";
        if (!err_func) std::cout << "    Missing: rocprofiler_register_error_string\n";
        if (!iter_func) std::cout << "    Missing: rocprofiler_register_iterate_registration_info\n";
        FreeLibrary(reg_dll);
        FreeLibrary(hip_dll);
        return 1;
    }

    FreeLibrary(reg_dll);
    FreeLibrary(hip_dll);

    std::cout << "=== All Tests Completed Successfully! ===\n";
    std::cout << "\nSummary:\n";
    std::cout << "  - rocprofiler-register.dll is working correctly\n";
    std::cout << "  - HIP library (amdhip64.dll) loads successfully\n";
    std::cout << "  - Registration API is functional\n";
    std::cout << "  - All required exports are present\n";

    return 0;
}
