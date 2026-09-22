Small benchmark utility for gpt-fast's all reduce. 

### How to run 
Out of box run (This will try various sequence lengths and dump perf results to terminal output)
```
torchrun --nproc_per_node=8 all_reduce.py 
```

To enable intra node all-reduce algorithms use:
```
ENABLE_INTRA_NODE_COMM=1 torchrun --nproc_per_node=8 python3 all_reduce.py
```

### Rocprof trace script
To create perfetto traces for each rank of each all reduce a bash script is provided. 
```
ENABLE_INTRA_NODE_COMM=1 bash trace_runs.sh
```

### Additional options:
The tensor size is dependent on sequence_length and dim supplied in gpt fast. There are 4 different all-reduce calls in gpt-fast at runtime:
- 1: [seq_len, dim]
- 2: [seq_len, 2, dim]
- 3: [1, dim]
- 4: [1, 2, dim]
```
--sequence_lengths (defaults to [50, 64, 128, 256, 512, 1024, 2048, 4096])
--dim (defaults to 6144)
--all_reduce (defaults to [0,1,2,3]) - Can be modified to only run a single all-reduce, mapping to the 4 all reduces listed above 
--tracing - Enables tracing mode to skip CPU timers in recording 
```

