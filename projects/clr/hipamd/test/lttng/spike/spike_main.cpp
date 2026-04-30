#include "spike_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Phase 0 spike entry point.
 *
 * Modes (controlled by argv[1]):
 *   (none)               -- emit 10 events immediately and exit. Used for
 *                           the basic build smoke test only -- DO NOT use
 *                           this mode to validate end-to-end capture; the
 *                           events fire before lttng-sessiond has finished
 *                           pushing the channel-enable to this app.
 *
 *   "trace" <emit_count> -- emit <emit_count> events ONCE PER SECOND for
 *                           up to 30 iterations or until killed. This is
 *                           the production-shape Step 7 mode: the
 *                           lttng-tools workflow can `lttng start` mid-run
 *                           and capture events from the next iteration
 *                           onward.
 *
 *   "loop" <emit_count> <iters>
 *                        -- like "trace" but with explicit iteration count.
 *
 *   "sleep" <secs>       -- emit 10 events ONCE then sleep for <secs>
 *                           seconds. Used by Step 9 (dlmopen,
 *                           RTLD_DEEPBIND) for `lttng list --userspace`
 *                           visibility checks. Events themselves may be
 *                           lost in this mode -- the test only checks
 *                           provider registration.
 *
 * Background on the timing: liblttng-ust spawns a listener thread at
 * library-init time that asynchronously connects to the sessiond app
 * socket. The first events emitted before that handshake completes have
 * no enabled channel attached, so they are silently dropped (this is by
 * design -- LTTng prioritizes producer non-blocking over completeness).
 * For a long-lived process this is a non-issue; for a sub-second test
 * binary we either insert a per-second emit loop (this file) or have
 * the test driver block on a signal/file before emitting.
 */
int main(int argc, char** argv) {
    if (argc >= 2 && (strcmp(argv[1], "trace") == 0 || strcmp(argv[1], "loop") == 0)) {
        int emit_count = (argc >= 3) ? atoi(argv[2]) : 10;
        int iters      = (strcmp(argv[1], "loop") == 0 && argc >= 4) ? atoi(argv[3]) : 30;
        if (emit_count <= 0) emit_count = 10;
        if (iters      <= 0) iters      = 30;

        fprintf(stderr,
                "[spike] trace mode: %d events x %d iters at 1 Hz (pid=%d)\n",
                emit_count, iters, (int)getpid());

        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < emit_count; ++i) {
                lttng_ust_tracepoint(rocm_spike, hello, it * emit_count + i,
                                     "from-spike");
            }
            sleep(1);
        }
        return 0;
    }

    /* Default + sleep modes. */
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
