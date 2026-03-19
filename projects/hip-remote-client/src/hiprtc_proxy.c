#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef int hiprtcResult;
typedef struct _hiprtcProgram* hiprtcProgram;

#define HIPRTC_SUCCESS 0

typedef hiprtcResult (*pfn_hiprtcCompileProgram)(hiprtcProgram, int, const char**);
typedef hiprtcResult (*pfn_hiprtcCreateProgram)(hiprtcProgram*, const char*, const char*,
                                                int, const char**, const char**);
typedef hiprtcResult (*pfn_hiprtcDestroyProgram)(hiprtcProgram*);
typedef hiprtcResult (*pfn_hiprtcGetCode)(hiprtcProgram, char*);
typedef hiprtcResult (*pfn_hiprtcGetCodeSize)(hiprtcProgram, size_t*);
typedef hiprtcResult (*pfn_hiprtcGetProgramLog)(hiprtcProgram, char*);
typedef hiprtcResult (*pfn_hiprtcGetProgramLogSize)(hiprtcProgram, size_t*);
typedef hiprtcResult (*pfn_hiprtcVersion)(int*, int*);
typedef hiprtcResult (*pfn_hiprtcAddNameExpression)(hiprtcProgram, const char*);
typedef hiprtcResult (*pfn_hiprtcGetLoweredName)(hiprtcProgram, const char*, const char**);
typedef const char*  (*pfn_hiprtcGetErrorString)(hiprtcResult);
typedef hiprtcResult (*pfn_hiprtcGetBitcode)(hiprtcProgram, char*);
typedef hiprtcResult (*pfn_hiprtcGetBitcodeSize)(hiprtcProgram, size_t*);

typedef void* hiprtcLinkState;
typedef int hiprtcJITInputType;
typedef int hiprtcJIT_option;

typedef hiprtcResult (*pfn_hiprtcLinkAddData)(hiprtcLinkState, hiprtcJITInputType, void*,
                                              size_t, const char*, int, hiprtcJIT_option*, void**);
typedef hiprtcResult (*pfn_hiprtcLinkAddFile)(hiprtcLinkState, hiprtcJITInputType, const char*,
                                              int, hiprtcJIT_option*, void**);
typedef hiprtcResult (*pfn_hiprtcLinkComplete)(hiprtcLinkState, void**, size_t*);
typedef hiprtcResult (*pfn_hiprtcLinkCreate)(int, hiprtcJIT_option*, void**, hiprtcLinkState*);
typedef hiprtcResult (*pfn_hiprtcLinkDestroy)(hiprtcLinkState);

static HMODULE g_real_hiprtc = NULL;
static char g_target_arch[64] = {0};

static HMODULE load_real(void) {
    if (g_real_hiprtc) return g_real_hiprtc;

    /* Load the real hipRTC DLL. _HIPRTC_REAL_PATH must point to the
     * original DLL (e.g. in _rocm_sdk_core/bin/). We never rename or
     * replace the original — this proxy is loaded under a different name
     * or from a different directory, and forwards calls to the real DLL. */
    const char* sdk_path = getenv("_HIPRTC_REAL_PATH");
    if (sdk_path && sdk_path[0]) {
        g_real_hiprtc = LoadLibraryA(sdk_path);
    }
    return g_real_hiprtc;
}

static void init_target(void) {
    const char* arch = getenv("HIP_REMOTE_TARGET_ARCH");
    if (arch && arch[0]) {
        strncpy(g_target_arch, arch, sizeof(g_target_arch) - 1);
    }
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL; (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_ATTACH) {
        init_target();
    }
    return TRUE;
}

#define GET_FUNC(name) \
    static pfn_##name fn = NULL; \
    if (!fn) { HMODULE m = load_real(); if (m) fn = (pfn_##name)GetProcAddress(m, #name); } \
    if (!fn) return 6 /* HIPRTC_ERROR_COMPILATION */

__declspec(dllexport) hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
                                                         int numOptions,
                                                         const char** options) {
    GET_FUNC(hiprtcCompileProgram);

    if (g_target_arch[0]) {
        /* Inject --offload-arch=<target> into the options */
        int newCount = numOptions + 1;
        const char** newOpts = (const char**)malloc(sizeof(const char*) * newCount);
        if (!newOpts) return fn(prog, numOptions, options);

        char archOpt[128];
        _snprintf(archOpt, sizeof(archOpt), "--offload-arch=%s", g_target_arch);

        for (int i = 0; i < numOptions; i++) {
            newOpts[i] = options[i];
        }
        newOpts[numOptions] = archOpt;

        hiprtcResult res = fn(prog, newCount, newOpts);
        free(newOpts);
        return res;
    }

    return fn(prog, numOptions, options);
}

__declspec(dllexport) hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src,
                                                        const char* name, int numHeaders,
                                                        const char** headers, const char** includeNames) {
    GET_FUNC(hiprtcCreateProgram);
    return fn(prog, src, name, numHeaders, headers, includeNames);
}

__declspec(dllexport) hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) {
    GET_FUNC(hiprtcDestroyProgram);
    return fn(prog);
}

__declspec(dllexport) hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code) {
    GET_FUNC(hiprtcGetCode);
    return fn(prog, code);
}

__declspec(dllexport) hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* size) {
    GET_FUNC(hiprtcGetCodeSize);
    return fn(prog, size);
}

__declspec(dllexport) hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log) {
    GET_FUNC(hiprtcGetProgramLog);
    return fn(prog, log);
}

__declspec(dllexport) hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* size) {
    GET_FUNC(hiprtcGetProgramLogSize);
    return fn(prog, size);
}

__declspec(dllexport) hiprtcResult hiprtcVersion(int* major, int* minor) {
    GET_FUNC(hiprtcVersion);
    return fn(major, minor);
}

__declspec(dllexport) hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name) {
    GET_FUNC(hiprtcAddNameExpression);
    return fn(prog, name);
}

__declspec(dllexport) hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char* name,
                                                         const char** lowered) {
    GET_FUNC(hiprtcGetLoweredName);
    return fn(prog, name, lowered);
}

__declspec(dllexport) const char* hiprtcGetErrorString(hiprtcResult result) {
    static pfn_hiprtcGetErrorString fn = NULL;
    if (!fn) { HMODULE m = load_real(); if (m) fn = (pfn_hiprtcGetErrorString)GetProcAddress(m, "hiprtcGetErrorString"); }
    if (!fn) return "unknown error";
    return fn(result);
}

__declspec(dllexport) hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char* code) {
    GET_FUNC(hiprtcGetBitcode);
    return fn(prog, code);
}

__declspec(dllexport) hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t* size) {
    GET_FUNC(hiprtcGetBitcodeSize);
    return fn(prog, size);
}

__declspec(dllexport) hiprtcResult hiprtcLinkAddData(hiprtcLinkState state, hiprtcJITInputType type,
                                                      void* data, size_t size, const char* name,
                                                      int numOpts, hiprtcJIT_option* opts, void** optVals) {
    GET_FUNC(hiprtcLinkAddData);
    return fn(state, type, data, size, name, numOpts, opts, optVals);
}

__declspec(dllexport) hiprtcResult hiprtcLinkAddFile(hiprtcLinkState state, hiprtcJITInputType type,
                                                      const char* path, int numOpts,
                                                      hiprtcJIT_option* opts, void** optVals) {
    GET_FUNC(hiprtcLinkAddFile);
    return fn(state, type, path, numOpts, opts, optVals);
}

__declspec(dllexport) hiprtcResult hiprtcLinkComplete(hiprtcLinkState state, void** bin, size_t* size) {
    GET_FUNC(hiprtcLinkComplete);
    return fn(state, bin, size);
}

__declspec(dllexport) hiprtcResult hiprtcLinkCreate(int numOpts, hiprtcJIT_option* opts,
                                                     void** optVals, hiprtcLinkState* state) {
    GET_FUNC(hiprtcLinkCreate);
    return fn(numOpts, opts, optVals, state);
}

__declspec(dllexport) hiprtcResult hiprtcLinkDestroy(hiprtcLinkState state) {
    GET_FUNC(hiprtcLinkDestroy);
    return fn(state);
}
