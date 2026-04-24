#include "spike_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Phase 0 spike entry point.
 *
 * Two modes:
 *   no args         -- emit 10 tracepoint events and exit (Step 7).
 *   "sleep" <secs>  -- emit 10 events, then sleep for <secs> seconds. Used
 *                      by the dlmopen / RTLD_DEEPBIND tests in Step 9 so
 *                      the host process stays around long enough for
 *                      `lttng list --userspace` to enumerate it.
 *
 * Events are flushed synchronously by lttng_ust_tracepoint(); the trailing
 * sleep exists only for the registration-visibility check, not for flush
 * correctness.
 */
int main(int argc, char** argv) {
    for (int i = 0; i < 10; ++i) {
        lttng_ust_tracepoint(rocm_spike, hello, i, "from-spike");
    }

    if (argc >= 3 && strcmp(argv[1], "sleep") == 0) {
        int secs = atoi(argv[2]);
        if (secs > 0) {
            fprintf(stderr, "[spike] emitted 10 events; sleeping %d s (pid=%d)\n",
                    secs, (int)getpid());
            sleep((unsigned)secs);
        }
    }
    return 0;
}
