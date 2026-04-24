#include "spike_tp.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    for (int i = 0; i < 10; ++i) {
        lttng_ust_tracepoint(rocm_spike, hello, i, "from-spike");
    }
    return 0;
}
