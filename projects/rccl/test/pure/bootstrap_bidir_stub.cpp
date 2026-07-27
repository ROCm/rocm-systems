// Standalone implementation of bootstrapBidirEnabled for CPU-only tests.
// Mirrors the production logic in src/bootstrap.cc but reads env vars
// directly (via getenv) instead of going through NCCL_PARAM infrastructure.
// This is safe because BootstrapBidirTests runs each case in a forked child
// process with a clean environment set by ProcessIsolatedTestRunner.

#include <cstdlib>
#include <cstdint>

static int64_t readEnvInt(const char* name, int64_t deftVal) {
    const char* s = getenv(name);
    if (s && s[0] != '\0') {
        char* end;
        int64_t v = strtoll(s, &end, 0);
        if (end != s) return v;
    }
    return deftVal;
}

static bool bootstrapNetEnabledEffective(int nranks) {
    int64_t v = readEnvInt("NCCL_OOB_NET_ENABLE", 0);
    if (v == 0) return false;
    if (v >= 1) return true;
    int64_t thr = readEnvInt("NCCL_BOOTSTRAP_BIDIR_THRESHOLD", 128);
    return thr > 0 && nranks >= (int)thr;
}

bool bootstrapBidirEnabled(int nranks, int kind) {
    if (nranks < 3) return false;
    bool netOn = bootstrapNetEnabledEffective(nranks);
    if (kind == 0) return (readEnvInt("NCCL_BOOTSTRAP_BIDIR_ALLGATHER", 1) != 0) && !netOn;
    if (kind == 1) {
        if (!netOn) return false;
        int64_t v = readEnvInt("NCCL_BOOTSTRAP_BIDIR_NET", 0);
        if (v == 0) return false;
        if (v >= 1) return true;
        int64_t thr = readEnvInt("NCCL_BOOTSTRAP_BIDIR_THRESHOLD", 128);
        return thr > 0 && nranks >= (int)thr;
    }
    return false;
}
