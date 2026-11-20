/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Windows regression test: rocDecode must fail cleanly when the TheRock VA-API
// DLLs (rocm_sysdeps_va_win32.dll / rocm_sysdeps_va.dll) cannot be found.
//
// rocdecode.dll locates the VA DLLs relative to its own location, then through
// %ROCM_PATH%, then through PATH. To make them undiscoverable, this test copies
// rocdecode.dll into an empty temporary directory, clears ROCM_PATH, removes
// every PATH entry that contains the VA DLL, and loads the copy at runtime
// (it does not link rocdecode). It then checks that rocDecGetDecoderCaps and
// rocDecCreateDecoder:
//   - return ROCDEC_NOT_INITIALIZED,
//   - do not let a C++ exception cross the C API,
//   - leave the create output handle untouched,
//   - behave the same on a repeated call (the VaContext singleton retries
//     construction after a failed attempt),
// and that no VA DLL was loaded into the process.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include "rocdecode/rocdecode.h"

namespace fs = std::filesystem;

namespace {

const wchar_t *kVaWin32Dll = L"rocm_sysdeps_va_win32.dll";
const wchar_t *kVaCoreDll = L"rocm_sysdeps_va.dll";

using GetDecoderCapsFn = rocDecStatus (*)(RocdecDecodeCaps *);
using CreateDecoderFn = rocDecStatus (*)(rocDecDecoderHandle *, RocDecoderCreateInfo *);

int g_failures = 0;

void Check(bool condition, const std::string &what) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << what << std::endl;
    if (!condition) {
        g_failures++;
    }
}

std::wstring GetEnvVar(const wchar_t *name) {
    std::wstring value;
    DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    while (size > value.size()) {
        value.resize(size);
        size = GetEnvironmentVariableW(name, value.data(), static_cast<DWORD>(value.size()));
    }
    value.resize(size);
    return value;
}

// Returns the rocdecode.dll to test: the command-line argument if given, else
// the installed DLL found at configure time, else %ROCM_PATH%\bin\rocdecode.dll.
fs::path FindRocDecodeDll(int argc, wchar_t **argv) {
    std::error_code ec;
    if (argc > 1) {
        return fs::path(argv[1]);
    }
#ifdef ROCDECODE_DLL_PATH
    fs::path configured(ROCDECODE_DLL_PATH);
    if (fs::is_regular_file(configured, ec)) {
        return configured;
    }
#endif
    std::wstring rocm_path = GetEnvVar(L"ROCM_PATH");
    if (!rocm_path.empty()) {
        fs::path from_env = fs::path(rocm_path) / "bin" / "rocdecode.dll";
        if (fs::is_regular_file(from_env, ec)) {
            return from_env;
        }
    }
    return {};
}

// Removes ROCM_PATH and every PATH entry that contains the VA DLL, so the only
// remaining way to reach the VA DLLs would be relative to rocdecode.dll itself.
// The installed rocdecode.dll's own directory is put first on PATH so its other
// imports (amdhip64, ...) still resolve for the relocated copy; that directory
// does not contain the VA DLLs (they live under lib\rocm_sysdeps\bin).
// _wputenv_s updates both the CRT's environment copy and the Win32 process
// environment, so the variables are hidden however the loader reads them.
void HideVaDlls(const fs::path &rocdecode_dir) {
    _wputenv_s(L"ROCM_PATH", L"");
    std::wstring path_env = rocdecode_dir.wstring() + L";" + GetEnvVar(L"PATH");
    std::wstring filtered;
    size_t start = 0;
    while (start <= path_env.size()) {
        size_t end = path_env.find(L';', start);
        if (end == std::wstring::npos) {
            end = path_env.size();
        }
        std::wstring entry = path_env.substr(start, end - start);
        std::wstring dir = entry;
        if (dir.size() >= 2 && dir.front() == L'"' && dir.back() == L'"') {
            dir = dir.substr(1, dir.size() - 2);
        }
        std::error_code ec;
        bool has_va_dll = !dir.empty() && fs::exists(fs::path(dir) / kVaWin32Dll, ec);
        if (!entry.empty() && !has_va_dll) {
            if (!filtered.empty()) {
                filtered += L';';
            }
            filtered += entry;
        }
        start = end + 1;
    }
    _wputenv_s(L"PATH", filtered.c_str());
}

RocDecoderCreateInfo MakeCreateInfo() {
    RocDecoderCreateInfo info = {};
    info.device_id = 0;
    info.width = 1920;
    info.height = 1080;
    info.num_decode_surfaces = 8;
    info.codec_type = rocDecVideoCodec_HEVC;
    info.chroma_format = rocDecVideoChromaFormat_420;
    info.bit_depth_minus_8 = 0;
    info.output_format = rocDecVideoSurfaceFormat_NV12;
    info.target_width = 1920;
    info.target_height = 1080;
    info.num_output_surfaces = 1;
    return info;
}

void RunChecks(GetDecoderCapsFn get_caps, CreateDecoderFn create_decoder) {
    for (int attempt = 1; attempt <= 2; attempt++) {
        std::string tag = " (attempt " + std::to_string(attempt) + ")";

        RocdecDecodeCaps caps = {};
        caps.device_id = 0;
        caps.codec_type = rocDecVideoCodec_HEVC;
        caps.chroma_format = rocDecVideoChromaFormat_420;
        rocDecStatus status = ROCDEC_SUCCESS;
        bool threw = false;
        try {
            status = get_caps(&caps);
        } catch (...) {
            threw = true;
        }
        Check(!threw, "rocDecGetDecoderCaps does not throw" + tag);
        Check(status == ROCDEC_NOT_INITIALIZED,
              "rocDecGetDecoderCaps returns ROCDEC_NOT_INITIALIZED" + tag + ", got " + std::to_string(status));

        RocDecoderCreateInfo create_info = MakeCreateInfo();
        rocDecDecoderHandle sentinel = reinterpret_cast<rocDecDecoderHandle>(static_cast<uintptr_t>(0xDEAD));
        rocDecDecoderHandle handle = sentinel;
        status = ROCDEC_SUCCESS;
        threw = false;
        try {
            status = create_decoder(&handle, &create_info);
        } catch (...) {
            threw = true;
        }
        Check(!threw, "rocDecCreateDecoder does not throw" + tag);
        Check(status == ROCDEC_NOT_INITIALIZED,
              "rocDecCreateDecoder returns ROCDEC_NOT_INITIALIZED" + tag + ", got " + std::to_string(status));
        Check(handle == sentinel, "rocDecCreateDecoder leaves the output handle untouched" + tag);
    }
}

// Copies rocdecode.dll into work_dir\bin, hides the VA DLLs, loads the copy and
// runs the checks. Returns false if the test could not be set up.
bool RunTest(const fs::path &installed_dll, const fs::path &work_dir) {
    std::error_code ec;
    fs::path bin_dir = work_dir / "bin";
    fs::create_directories(bin_dir, ec);
    fs::path dll_copy = bin_dir / "rocdecode.dll";
    if (!fs::copy_file(installed_dll, dll_copy, fs::copy_options::overwrite_existing, ec)) {
        std::cerr << "Test Failed! Cannot copy " << installed_dll.u8string() << ": " << ec.message() << std::endl;
        return false;
    }

    HideVaDlls(installed_dll.parent_path());
    // Not a lookup location for rocdecode, but keep the working directory clean too.
    SetCurrentDirectoryW(work_dir.c_str());

    // Standard search order so rocdecode.dll's own imports (amdhip64, d3d12, ...) still resolve.
    HMODULE module = LoadLibraryW(dll_copy.c_str());
    if (!module) {
        std::cerr << "Test Failed! LoadLibraryW('" << dll_copy.u8string() << "') failed with error "
                  << GetLastError() << std::endl;
        return false;
    }
    auto get_caps = reinterpret_cast<GetDecoderCapsFn>(GetProcAddress(module, "rocDecGetDecoderCaps"));
    auto create_decoder = reinterpret_cast<CreateDecoderFn>(GetProcAddress(module, "rocDecCreateDecoder"));
    if (!get_caps || !create_decoder) {
        std::cerr << "Test Failed! rocDecGetDecoderCaps/rocDecCreateDecoder not exported." << std::endl;
        FreeLibrary(module);
        return false;
    }

    std::cout << "info: Testing rocDecode with the VA-API DLLs hidden, using " << dll_copy.u8string() << std::endl;
    RunChecks(get_caps, create_decoder);
    Check(GetModuleHandleW(kVaWin32Dll) == nullptr && GetModuleHandleW(kVaCoreDll) == nullptr,
          "no VA-API DLL was loaded into the process");
    FreeLibrary(module);
    return true;
}

} // namespace

int wmain(int argc, wchar_t **argv) {
    std::error_code ec;
    fs::path installed_dll = FindRocDecodeDll(argc, argv);
    if (installed_dll.empty()) {
        std::cerr << "Test Failed! Cannot locate rocdecode.dll; pass its path or set ROCM_PATH to the ROCm installation prefix."
                  << std::endl;
        return EXIT_FAILURE;
    }
    // Absolute, because the working directory changes before the copy is loaded.
    installed_dll = fs::absolute(installed_dll, ec);

    // An empty directory with no lib\rocm_sysdeps\bin next to it.
    fs::path temp_dir = fs::temp_directory_path(ec);
    if (ec || temp_dir.empty()) {
        std::cerr << "Test Failed! Cannot determine the temporary directory: " << ec.message() << std::endl;
        return EXIT_FAILURE;
    }
    fs::path work_dir = fs::absolute(temp_dir, ec) / (L"rocdecode_missing_va_" + std::to_wstring(GetCurrentProcessId()));
    fs::remove_all(work_dir, ec);

    bool ran = RunTest(installed_dll, work_dir);

    // Clean up on every path; leave the directory first so it can be removed.
    SetCurrentDirectoryW(temp_dir.c_str());
    fs::remove_all(work_dir, ec);

    if (!ran) {
        return EXIT_FAILURE;
    }
    if (g_failures) {
        std::cout << "Test Failed! " << g_failures << " check(s) failed." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Test Passed!" << std::endl;
    return EXIT_SUCCESS;
}
