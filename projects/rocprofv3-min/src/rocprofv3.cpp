// rocprofv3.exe -- minimal launcher.
//
// Parses --hip-trace / --kernel-trace / --memory-copy-trace / -d <dir>,
// sets corresponding environment variables that rocprofv3-min.dll reads,
// sets ROCPROFILER_REGISTER_LIBRARY to point at rocprofv3-min.dll, then
// CreateProcess's the user app.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf, n);
    size_t p = s.find_last_of("\\/");
    return p == std::string::npos ? std::string(".") : s.substr(0, p);
}

static void usage() {
    std::fprintf(stderr,
        "Usage: rocprofv3 [--hip-trace] [--kernel-trace] [--memory-copy-trace]\n"
        "                 [-d <output_dir>] -- <app> [args...]\n");
}

int main(int argc, char** argv) {
    bool hip_trace = false, kernel_trace = false, memcopy_trace = false;
    std::string out_dir = ".";
    int i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hip-trace") hip_trace = true;
        else if (a == "--kernel-trace") kernel_trace = true;
        else if (a == "--memory-copy-trace") memcopy_trace = true;
        else if (a == "-d" && i + 1 < argc) { out_dir = argv[++i]; }
        else if (a == "--") { ++i; break; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { break; }
    }
    if (i >= argc) { usage(); return 1; }

    std::string dll = exe_dir() + "\\rocprofv3-min.dll";

    SetEnvironmentVariableA("ROCPROFILER_REGISTER_LIBRARY", dll.c_str());
    if (hip_trace)     SetEnvironmentVariableA("ROCPROFV3_HIP_TRACE", "1");
    if (kernel_trace)  SetEnvironmentVariableA("ROCPROFV3_KERNEL_TRACE", "1");
    if (memcopy_trace) SetEnvironmentVariableA("ROCPROFV3_MEMORY_COPY_TRACE", "1");
    SetEnvironmentVariableA("ROCPROFV3_OUTPUT_DIR", out_dir.c_str());

    // Build command-line for child.
    std::string cmd;
    for (int j = i; j < argc; ++j) {
        if (j > i) cmd += ' ';
        bool has_space = std::strchr(argv[j], ' ') != nullptr;
        if (has_space) cmd += '"';
        cmd += argv[j];
        if (has_space) cmd += '"';
    }

    std::fprintf(stderr,
        "[rocprofv3] launching: %s\n"
        "[rocprofv3] interceptor: %s\n"
        "[rocprofv3] output_dir: %s\n"
        "[rocprofv3] hip=%d kernel=%d memcopy=%d\n",
        cmd.c_str(), dll.c_str(), out_dir.c_str(),
        (int)hip_trace, (int)kernel_trace, (int)memcopy_trace);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<char> mut(cmd.begin(), cmd.end()); mut.push_back('\0');
    if (!CreateProcessA(nullptr, mut.data(), nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &si, &pi)) {
        std::fprintf(stderr, "[rocprofv3] CreateProcess failed: %lu\n", GetLastError());
        return 2;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)exit_code;
}
