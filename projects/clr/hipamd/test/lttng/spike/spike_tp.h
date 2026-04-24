#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_spike

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "./spike_tp.h"

#if !defined(_ROCM_SPIKE_TP_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define _ROCM_SPIKE_TP_H

#include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(
    rocm_spike,
    hello,
    LTTNG_UST_TP_ARGS(int, value, const char*, name),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(int, value, value)
        lttng_ust_field_string(name, name)
    )
)

#endif /* _ROCM_SPIKE_TP_H */

#include <lttng/tracepoint-event.h>
