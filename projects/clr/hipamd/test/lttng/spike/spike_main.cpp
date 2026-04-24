#include "spike_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Phase 0 spike entry point.
 *
 * Modes (controlled by argv[1]):
 *   (none)               -- emit 10 events immediately and exit. Used by
 *                           the basic Step 7 sanity build.
 *   "trace" <emit_count> -- wait LTTNG_UST_REGISTER_TIMEOUT (default 3 s)
 *                           via an explicit pre-emit sleep so this app's
 *                           registration with lttng-sessiond is fully
 *                           processed and any enabled rocm_spike:hello
 *                           channel is attached BEFORE we emit the events.
 *                           Then emit <emit_count> events, then a short
 *                           post-emit sleep to give the consumer thread
 *                           time to flush sub-buffers to disk.
 *                           This is the production-shape Step 7 mode.
 *   "sleep" <secs>       -- emit 10 events, then sleep for <secs> seconds.
 *                           Used by the dlmopen / RTLD_DEEPBIND tests in
 *                           Step 9 so the host process stays around long
 *                           enough for `lttng list --userspace` to
 *                           enumerate it.
 *
 * Background on the timing: liblttng-ust spawns a listener thread at
 * library-init time that asynchronously connects to the sessiond app
 * socket. The first events emitted before that handshake completes have
 * no enabled channel attached, so they are silently dropped (this is by
 * design -- LTTng prioritizes producer non-blocking over completeness).
 * For a long-lived process this is a non-issue; for a sub-second test
 * binary we have to insert an explicit pre-emit sleep, OR rely on the
 * LTTNG_UST_REGISTER_TIMEOUT env var which makes liblttng-ust block for
 * up to N ms during library init.
 */
int main(int argc, char** argv) {
    int emit_count = 10;
    bool wait_for_register = false;

    if (argc >= 2 && strcmp(argv[1], "trace") == 0) {
        wait_for_register = true;
        if (argc >= 3) emit_count = atoi(argv[2]);
        if (emit_count <= 0) emit_count = 10;
    }

    if (wait_for_register) {
        /* Belt-and-suspenders: LTTNG_UST_REGISTER_TIMEOUT in the env should
         * already make library-init block until sessiond responds, but we
         * sleep an extra second here to also cover the case where the
         * channel-enable propagation from sessiond->app lags. Cheap on a
         * test binary. */
        sleep(1);
    }

    for (int i = 0; i < emit_count; ++i) {
        lttng_ust_tracepoint(rocm_spike, hello, i, "from-spike");
    }

    if (wait_for_register) {
        /* Allow the consumer thread to flush sub-buffers before the process
         * exits and lttng_ust's destructor tears down the per-CPU buffers. */
        sleep(1);
    }

    if (argc >= 3 && strcmp(argv[1], "sleep") == 0) {
        int secs = atoi(argv[2]);
        if (secs > 0) {
            fprintf(stderr, "[spike] emitted %d events; sleeping %d s (pid=%d)\n",
                    emit_count, secs, (int)getpid());
            sleep((unsigned)secs);
        }
    }
    return 0;
}
