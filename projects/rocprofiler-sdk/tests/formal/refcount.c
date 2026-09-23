// CBMC harness: faithful model of QueueController::serialization_refcount and
// QueueController::update_serialization().
//
// Source of truth:
//   projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue_controller.hpp:155  (struct)
//   projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue_controller.cpp:728  (update)
//
// Build with -DALLOW_UNBALANCED to drop the "callers pair their calls" assumption.

#include <assert.h>
#include <stdint.h>

#ifndef NAGENTS
#define NAGENTS 2
#endif
#ifndef NCLIENTS
#define NCLIENTS 3
#endif
#ifndef NOPS
#define NOPS 5
#endif

int  nondet_int(void);
_Bool nondet_bool(void);

/* ---------- serialization_refcount ---------- */
static int64_t st_all;
static int64_t st_per_agent[NAGENTS];

static int64_t rc_count(int a) { return st_all + st_per_agent[a]; }
static int     rc_enabled(int a) { return rc_count(a) > 0; }

/* ---------- serializer, toggled only on a transition ---------- */
static int ser_enabled[NAGENTS];

/* ---------- update_serialization(agents, enable) ---------- */
static void
update_serialization(int use_all, unsigned mask, int enable)
{
    int was[NAGENTS];
    for(int a = 0; a < NAGENTS; ++a)
        was[a] = rc_enabled(a);

    if(use_all)
    {
        if(enable)
            ++st_all;
        else if(st_all > 0)
            --st_all;
    }
    else
    {
        for(int a = 0; a < NAGENTS; ++a)
        {
            if(mask & (1u << a))
            {
                if(enable)
                    ++st_per_agent[a];
                else if(st_per_agent[a] > 0)
                    --st_per_agent[a];
            }
        }
    }

    for(int a = 0; a < NAGENTS; ++a)
    {
        int now = rc_enabled(a);
        if(was[a] == now) continue;
        ser_enabled[a] = now;
    }
}

int
main(void)
{
    st_all = 0;
    for(int a = 0; a < NAGENTS; ++a)
    {
        st_per_agent[a] = 0;
        ser_enabled[a]  = 0;
    }

    int      holds[NCLIENTS];
    int      c_use_all[NCLIENTS];
    unsigned c_mask[NCLIENTS];

    for(int c = 0; c < NCLIENTS; ++c)
    {
        holds[c]     = 0;
        c_use_all[c] = nondet_bool();
        unsigned m   = (unsigned) nondet_int();
        __CPROVER_assume(m < (1u << NAGENTS));
        // An explicitly empty agent set is the "all agents" sentinel, so a
        // per-agent client always names at least one agent.
        __CPROVER_assume(c_use_all[c] || m != 0);
        c_mask[c] = m;
    }

    for(int step = 0; step < NOPS; ++step)
    {
        int c = nondet_int();
        __CPROVER_assume(c >= 0 && c < NCLIENTS);
        int enable = nondet_bool();

#ifndef ALLOW_UNBALANCED
        // Services pair their calls: they only release what they acquired.
        __CPROVER_assume(enable ? !holds[c] : holds[c]);
#endif

        update_serialization(c_use_all[c], c_mask[c], enable);
        holds[c] = enable;

        // P1: the serializer mirrors the refcount exactly (no drift, no spurious toggle).
        for(int a = 0; a < NAGENTS; ++a)
            assert(ser_enabled[a] == rc_enabled(a));

        // P2: counts never go negative.
        assert(st_all >= 0);
        for(int a = 0; a < NAGENTS; ++a)
            assert(st_per_agent[a] >= 0);

        // P3 (safety): if any client still holds a claim covering agent a, serialization
        // for a must still be on. This is the property a profiling service depends on.
        for(int a = 0; a < NAGENTS; ++a)
        {
            int needed = 0;
            for(int cc = 0; cc < NCLIENTS; ++cc)
                if(holds[cc] && (c_use_all[cc] || (c_mask[cc] & (1u << a)))) needed = 1;
            if(needed) assert(ser_enabled[a]);
        }
    }
    return 0;
}
