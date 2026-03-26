# Topology Visualizer
Topology Visualizer extracts topology information from RCCL log file and presents graphically. Less than optimal connections between GPUs and nodes are highlighted in red for easy identification.

## Requirements
Following packages are required to run Topology Visualizer:
1. gawk
2. graphviz

## Usage
Topology Visualizer accepts both RCCL log files or simulator output, i.e. [Topology Explorer](https://github.com/ROCm/rccl/tree/master/tools/topo_expl "Topology Explorer").

RCCL logs needs to be collected with NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=INIT,GRAPH environmental variables. Example command line:
```shell
mpirun -np 4 -host rocm-framework-1,rocm-framework-3,rocm-framework-5,rocm-framework-6 \
  -env HSA_FORCE_FINE_GRAIN_PCIE 1 -env NCCL_DEBUG INFO -env NCCL_DEBUG_SUBSYS INIT,GRAPH \
  ~/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8 | tee ~/4_nodes.log

./topo_visual.sh -i 4_nodes.log
```

## Command Line Options

```
Usage: topo_visual.sh -i input_filename [-f format] [-d dpi] [-l]

Options:
  -i input_filename   Input log file (required)
  -f format           Output format: svg (default), png, pdf
                      SVG recommended for large topologies (scalable, no blur)
  -d dpi              DPI for PNG output (default: 300, use 600+ for large graphs)
  -l                  Large graph mode: optimizes layout for 64+ GPUs
```

### Examples

```shell
# Generate SVG (recommended for large topologies - infinite zoom without blur)
./topo_visual.sh -i nccl_log.txt

# Generate high-DPI PNG for 256 GPU topology
./topo_visual.sh -i nccl_log.txt -f png -d 600

# SVG with large graph layout optimizations
./topo_visual.sh -i nccl_log.txt -f svg -l

# Generate PDF (also vector format, no blur)
./topo_visual.sh -i nccl_log.txt -f pdf
```

## Output Formats

| Format | Type | Best For | Zoom Quality |
|--------|------|----------|--------------|
| **SVG** (default) | Vector | Large topologies (64+ GPUs), web viewing | Perfect at any zoom |
| **PNG** | Raster | Documentation, sharing | Depends on DPI |
| **PDF** | Vector | Printing, documentation | Perfect at any zoom |

### Recommendations for Large Topologies (64+ GPUs)

For topologies with many GPUs (e.g., 256 GPUs across multiple nodes):

1. **Use SVG format** (default): Vector graphics scale infinitely without blur
2. **Use `-l` flag**: Enables layout optimizations for large graphs
3. **For PNG**: Use high DPI (600+) with `-d 600` or higher

The visualizer automatically detects large topologies and adjusts:
- Node sizes and font sizes for readability
- Graph layout parameters to prevent overlapping
- Edge rendering for clearer connections

## Legend

Solid lines: connections over P2P or shared memory

Dashed lines: connections over network

Green: P2P connections, network connections with GPU RDMA

Red: Connections over shared memory or without GPU RDMA

## Example Output
![image info](./4_nodes.log.png)

## Copyright
All source code and accompanying documentation are copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
