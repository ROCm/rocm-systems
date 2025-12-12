/*
 * amdgpu_pmu_trace.h - Tracepoint definitions for AMDGPU PMU
 *
 * This header defines tracepoints for GPU performance counter sampling,
 * allowing perf record to capture counter samples as trace events.
 *
 * Usage:
 *   perf record -e amdgpu_pmu:counter_sample -a ./app
 *   perf script
 */

#undef TRACE_SYSTEM
#define TRACE_SYSTEM amdgpu_pmu

#if !defined(_AMDGPU_PMU_TRACE_H) || defined(TRACE_HEADER_MULTI_READ)
#define _AMDGPU_PMU_TRACE_H

#include <linux/tracepoint.h>

/**
 * counter_sample - Emitted when a GPU counter is sampled
 * @counter_id: Numeric counter identifier (from counter_id_t enum)
 * @counter_name: Human-readable counter name (e.g., "SQ_WAVES")
 * @value: Current counter value
 * @delta: Change since last sample (for rate calculation)
 * @timestamp_ns: High-precision timestamp (ktime_get_ns())
 *
 * This tracepoint is emitted by the timer handler for each active sampling
 * event. It provides full visibility into GPU counter behavior over time.
 *
 * Example output:
 *   test_32_waves-1234 [000] 1000.123: amdgpu_pmu:counter_sample: \
 *       id=30 name=SQ_WAVES value=64 delta=32 ts=1000123456789
 */
TRACE_EVENT(counter_sample,

	    TP_PROTO(u32 counter_id, const char *counter_name, u64 value, u64 delta,
		     u64 timestamp_ns),

	    TP_ARGS(counter_id, counter_name, value, delta, timestamp_ns),

	    TP_STRUCT__entry(__field(u32, counter_id) __string(counter_name, counter_name)
				     __field(u64, value) __field(u64, delta)
					     __field(u64, timestamp_ns)),

	    TP_fast_assign(__entry->counter_id = counter_id; __assign_str(counter_name);
			   __entry->value = value; __entry->delta = delta;
			   __entry->timestamp_ns = timestamp_ns;),

	    TP_printk("id=%u name=%s value=%llu delta=%llu ts=%llu", __entry->counter_id,
		      __get_str(counter_name), __entry->value, __entry->delta,
		      __entry->timestamp_ns));

/**
 * kernel_dispatch - GPU kernel dispatch event from rocprofv3
 * Note: Limited to 12 args due to kernel tracepoint constraints
 */
TRACE_EVENT(kernel_dispatch,
    TP_PROTO(u64 dispatch_id, u64 correlation_id, u64 kernel_id,
             u64 start_ts, u64 end_ts, u32 agent_id, u32 queue_id,
             u32 grid_x, u32 grid_y, u32 grid_z,
             u32 workgroup_size),

    TP_ARGS(dispatch_id, correlation_id, kernel_id, start_ts, end_ts,
            agent_id, queue_id, grid_x, grid_y, grid_z, workgroup_size),

    TP_STRUCT__entry(
        __field(u64, dispatch_id)
        __field(u64, correlation_id)
        __field(u64, kernel_id)
        __field(u64, start_ts)
        __field(u64, end_ts)
        __field(u32, agent_id)
        __field(u32, queue_id)
        __field(u32, grid_x)
        __field(u32, grid_y)
        __field(u32, grid_z)
        __field(u32, workgroup_size)
    ),

    TP_fast_assign(
        __entry->dispatch_id = dispatch_id;
        __entry->correlation_id = correlation_id;
        __entry->kernel_id = kernel_id;
        __entry->start_ts = start_ts;
        __entry->end_ts = end_ts;
        __entry->agent_id = agent_id;
        __entry->queue_id = queue_id;
        __entry->grid_x = grid_x;
        __entry->grid_y = grid_y;
        __entry->grid_z = grid_z;
        __entry->workgroup_size = workgroup_size;
    ),

    TP_printk("dispatch=%llu corr=%llu kernel=%llu start=%llu end=%llu "
              "agent=%u queue=%u grid=%u,%u,%u wg_size=%u",
              __entry->dispatch_id, __entry->correlation_id, __entry->kernel_id,
              __entry->start_ts, __entry->end_ts, __entry->agent_id,
              __entry->queue_id, __entry->grid_x, __entry->grid_y, __entry->grid_z,
              __entry->workgroup_size)
);

/**
 * hsa_api - HSA API call event from rocprofv3
 */
TRACE_EVENT(hsa_api,
    TP_PROTO(u32 kind, u32 operation, u64 correlation_id,
             u64 start_ts, u64 end_ts, u32 thread_id, const char *operation_name),

    TP_ARGS(kind, operation, correlation_id, start_ts, end_ts, thread_id, operation_name),

    TP_STRUCT__entry(
        __field(u32, kind)
        __field(u32, operation)
        __field(u64, correlation_id)
        __field(u64, start_ts)
        __field(u64, end_ts)
        __field(u32, thread_id)
        __string(operation_name, operation_name)
    ),

    TP_fast_assign(
        __entry->kind = kind;
        __entry->operation = operation;
        __entry->correlation_id = correlation_id;
        __entry->start_ts = start_ts;
        __entry->end_ts = end_ts;
        __entry->thread_id = thread_id;
        __assign_str(operation_name);
    ),

    TP_printk("name=%s kind=%u op=%u corr=%llu start=%llu end=%llu tid=%u",
              __get_str(operation_name), __entry->kind, __entry->operation, __entry->correlation_id,
              __entry->start_ts, __entry->end_ts, __entry->thread_id)
);

/**
 * hip_api - HIP API call event from rocprofv3
 */
TRACE_EVENT(hip_api,
    TP_PROTO(u32 kind, u32 operation, u64 correlation_id,
             u64 start_ts, u64 end_ts, u32 thread_id, const char *operation_name),

    TP_ARGS(kind, operation, correlation_id, start_ts, end_ts, thread_id, operation_name),

    TP_STRUCT__entry(
        __field(u32, kind)
        __field(u32, operation)
        __field(u64, correlation_id)
        __field(u64, start_ts)
        __field(u64, end_ts)
        __field(u32, thread_id)
        __string(operation_name, operation_name)
    ),

    TP_fast_assign(
        __entry->kind = kind;
        __entry->operation = operation;
        __entry->correlation_id = correlation_id;
        __entry->start_ts = start_ts;
        __entry->end_ts = end_ts;
        __entry->thread_id = thread_id;
        __assign_str(operation_name);
    ),

    TP_printk("name=%s kind=%u op=%u corr=%llu start=%llu end=%llu tid=%u",
              __get_str(operation_name), __entry->kind, __entry->operation, __entry->correlation_id,
              __entry->start_ts, __entry->end_ts, __entry->thread_id)
);

#endif /* _AMDGPU_PMU_TRACE_H */

/* This part must be outside protection */
#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH ../../src
#undef TRACE_INCLUDE_FILE
#define TRACE_INCLUDE_FILE amdgpu_pmu_trace
#include <trace/define_trace.h>
