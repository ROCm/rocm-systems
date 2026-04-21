// htl_loader.cpp — ctor/dtor; resolves hipRegisterTracerCallback via
// dlsym(RTLD_DEFAULT, ...); env-var parsing; owns the global Writer.
#include "htl_callback.hpp"
#include "htl_writer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace htl {
extern Writer* g_writer;
extern std::atomic<bool> g_capture_hip_ops;
extern std::atomic<bool> g_capture_hip_api;
}  // namespace htl

namespace {

using register_fn_t = void (*)(int (*)(uint32_t, uint32_t, void*));

htl::Writer*  s_writer = nullptr;
register_fn_t s_register = nullptr;

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
}

std::string env_str(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string(dflt);
}

}  // namespace

extern "C" __attribute__((constructor))
void htl_init() {
    htl::g_capture_hip_ops.store(true,                  std::memory_order_release);
    htl::g_capture_hip_api.store(env_truthy("HTL_TRACE_API"), std::memory_order_release);

    const std::string out = env_str("HTL_OUTPUT_FILE", "./hiptrace.bin");
    s_writer = new htl::Writer();
    if (!s_writer->start(out)) {
        std::fprintf(stderr, "[hip-trace-lite] Writer::start failed; tracing disabled\n");
        delete s_writer;
        s_writer = nullptr;
        return;
    }
    htl::g_writer = s_writer;

    s_register = reinterpret_cast<register_fn_t>(
        dlsym(RTLD_DEFAULT, "hipRegisterTracerCallback"));
    if (!s_register) {
        std::fprintf(stderr,
            "[hip-trace-lite] dlsym(hipRegisterTracerCallback) failed: %s\n",
            dlerror());
        // Writer stays open in case the symbol shows up later — but we cannot
        // register, so no records will arrive.
        return;
    }
    s_register(&htl::htl_tracer_callback);
    std::fprintf(stderr, "[hip-trace-lite] registered, output=%s api=%d\n",
                 out.c_str(),
                 htl::g_capture_hip_api.load(std::memory_order_relaxed) ? 1 : 0);
}

extern "C" __attribute__((destructor))
void htl_fini() {
    if (s_register) s_register(nullptr);  // detach
    if (s_writer) {
        s_writer->stop();
        std::fprintf(stderr,
            "[hip-trace-lite] shutdown: %llu records written, %llu dropped\n",
            (unsigned long long)s_writer->records_written(),
            (unsigned long long)s_writer->records_dropped());
        delete s_writer;
        s_writer = nullptr;
        htl::g_writer = nullptr;
    }
}
