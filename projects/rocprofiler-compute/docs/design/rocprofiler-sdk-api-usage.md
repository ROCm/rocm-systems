# rocprofiler-sdk APIs used by rocprof-compute

These are the rocprofiler-sdk APIs that rocprof-compute uses to collect
profiling data. Its native tool (`librocprofiler-compute-tool.so`) is loaded
into the profiled application with `LD_PRELOAD` and calls them. SDK changes
must keep these APIs working, or offer a replacement.

## Counter collection

- `rocprofiler_configure_callback_dispatch_counting_service`, configured in
  `tool_init` when counters are requested.
  - Counters must work when this service is configured before HSA loads. Today
    this selects the legacy intercept path, because queue interposition (the
    default) cannot collect counters. The tool does not set
    `ROCPROFILER_QUEUE_INTERPOSITION`, so the SDK must pick a working mode.
- The dispatch callback chooses a counter config for each dispatch:
  - It sets no config for dispatches excluded by the kernel name filter or the
    dispatch range filter.
  - Iteration multiplexing gives successive dispatches of one kernel different
    configs, in round robin.
  - Configs are created inside the callback, on the first dispatch per agent.
- Fields read from `dispatch_info`: `dispatch_id`, `kernel_id`, `agent_id`,
  `queue_id`, `grid_size`, `workgroup_size`, `group_segment_size`.
- Setup: `rocprofiler_iterate_agent_supported_counters`,
  `rocprofiler_query_counter_info` (`ROCPROFILER_COUNTER_INFO_VERSION_0`, name
  only), `rocprofiler_create_counter_config`.
- Record callback: `rocprofiler_query_record_counter_id` on each record `id`,
  plus `counter_value`.
- Counter values must belong to exactly one dispatch.

## Kernel dispatch tracing

Gives the timing and launch parameters of each kernel dispatch.

- API: `rocprofiler_configure_callback_tracing_service` with
  `ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH`, operation
  `ROCPROFILER_KERNEL_DISPATCH_COMPLETE`.
- Fields: `start_timestamp`, `end_timestamp`, `correlation_id.internal`. From
  `dispatch_info`: `dispatch_id`, `kernel_id`, `agent_id`, `grid_size`,
  `workgroup_size`, `group_segment_size`, `private_segment_size`.

## Code object tracing

Gives the name and register counts of each kernel, and the code objects that
PC sampling decodes.

- API: `rocprofiler_configure_callback_tracing_service` with
  `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`, load phase.
- `ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER` fields: `kernel_id`,
  `kernel_name`, `code_object_id`, `sgpr_count`, `arch_vgpr_count`,
  `accum_vgpr_count`.
- `ROCPROFILER_CODE_OBJECT_LOAD` fields: `code_object_id`, `storage_type`
  (`FILE` or `MEMORY`), `uri`, `memory_base`, `memory_size`, `load_base`,
  `load_size`.
- A kernel symbol must be registered before the first dispatch of that kernel,
  so counter collection can be filtered by kernel name.

## GPU agents

Identifies the GPU that ran each dispatch.

- API: `rocprofiler_query_available_agents` with
  `ROCPROFILER_AGENT_INFO_VERSION_0`.
- Fields: `type`, `id`, `node_id`, `name`, `product_name`.

## PC sampling

Maps each sampled program counter to its instruction and stall reason.

- APIs: `rocprofiler_query_pc_sampling_agent_configurations`,
  `rocprofiler_configure_pc_sampling_service`, `rocprofiler_create_buffer`,
  `rocprofiler_flush_buffer`.
- Methods: host trap with unit time, and stochastic with unit cycles.
- Records: `rocprofiler_pc_sampling_record_host_trap_v0_t` and
  `rocprofiler_pc_sampling_record_stochastic_v0_t`. Fields:
  `pc.code_object_id`, `pc.code_object_offset`, `dispatch_id`, `wave_issued`,
  `inst_type`, and the not-issued reason from the snapshot.
- Name APIs: `rocprofiler_get_pc_sampling_instruction_type_name`,
  `rocprofiler_get_pc_sampling_instruction_not_issued_reason_name`.
- Code object decoding API (`rocprofiler-sdk/cxx/codeobj/code_printing.hpp`):
  - `codeobj::disassembly::CodeobjAddressTranslate`: `addDecoder` from a file
    URI or from memory, `getSymbolMap` (`name`, `vaddr`, `faddr`, `mem_size`)
    and `get` (`inst`, `comment`, `faddr`, `size`).
  - `codeobj::disassembly::Instruction::separator`, which splits the source
    frames in an instruction comment.

## ROCTX markers

Attributes kernels to the application ranges that launched them.

- API: `rocprofiler_configure_callback_tracing_service` with
  `ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API`, operations `roctxRangePushA`,
  `roctxRangePop`, `roctxRangeStartA`, `roctxRangeStop`.
- Library: `librocprofiler-sdk-roctx.so` (`rocprofiler-sdk-roctx/roctx.h`).
