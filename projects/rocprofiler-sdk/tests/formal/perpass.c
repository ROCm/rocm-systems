// CBMC harness: per-pass service isolation under kernel replay.
//
// Formalises the claim "kernel replay runs multiple services, each one in its own pass".
//
// Source of truth:
//   kernel_replay/local_context.hpp          sticky, loop-scoped thread-local override map
//   counters/dispatch_handlers.cpp:87        enabled = enabled && *ov
//   spm/dispatch_handlers.cpp:150            enabled = enabled && *ov
//   thread_trace/core.cpp:508                if(ov && !*ov) skip trace
//   pc_sampling/queue_hooks.cpp              no override consult at all
//
// SERVICES: 0=counters 1=spm 2=thread_trace 3=pc_sampling

#include <assert.h>

#define NSVC 4
#define PCS 3

#ifndef NPASS
#define NPASS 3
#endif

int   nondet_int(void);
_Bool nondet_bool(void);

static int g_enabled[NSVC];  // globally started contexts

// Thread-local override map: -1 = no entry, 0 = forced off, 1 = forced on.
// Sticky: an entry persists across passes until overwritten.
static int ov[NSVC];

// Effective collection state for a dispatch in the current pass.
static int
effective(int s)
{
    if(s == PCS) return g_enabled[s];       // PC sampling ignores the override
    if(ov[s] == -1) return g_enabled[s];    // no override recorded -> global state
    return g_enabled[s] && ov[s];           // AND semantics: local cannot promote
}

int
main(void)
{
    // A tool starts every service it wants to multiplex across passes.
    for(int s = 0; s < NSVC; ++s)
        g_enabled[s] = nondet_bool();

    for(int s = 0; s < NSVC; ++s)
        ov[s] = -1;

    // The tool's intended schedule: pass p runs service want[p].
    int want[NPASS];
    for(int p = 0; p < NPASS; ++p)
    {
        int w = nondet_int();
        __CPROVER_assume(w >= 0 && w < NSVC);
        want[p] = w;
    }

    for(int p = 0; p < NPASS; ++p)
    {
        // PASS PHASE_ENTER: the tool records overrides for this pass.
        // It explicitly names every service, which is what a correct tool must do
        // given the map is sticky.
        for(int s = 0; s < NSVC; ++s)
            ov[s] = (s == want[p]) ? 1 : 0;

        // ---- GOAL: among the override-aware services, at most one collects ----
        int n_aware = 0;
        for(int s = 0; s < NSVC; ++s)
            if(s != PCS && effective(s)) n_aware++;
        assert(n_aware <= 1);

        // ---- GOAL (strong form): at most one service collects, period ----
        // This is the property that actually matters for hardware contention.
        int n_all = 0;
        for(int s = 0; s < NSVC; ++s)
            if(effective(s)) n_all++;
        assert(n_all <= 1);
    }

    return 0;
}
