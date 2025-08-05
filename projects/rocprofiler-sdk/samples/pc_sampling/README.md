# PC sampling service

## Services

- PC sampling stochastic method

## Properties

- Iterate through all gpu agents that supports PC sampling.
- Iterate through the supported configuration for that agent.
- The `configure_pc_sampling_prefer_stochastic` function is responsible for configuring PC sampling on a given GPU agent. It attempts to select a stochastic sampling configuration if available, falling back to a host-trap configuration otherwise.
- `rocprofiler_pc_sampling_callback` function processes PC sampling records delivered by the profiler. It validates the records, determines their type, and delegates the printing of their details to the appropriate print_sample function.
