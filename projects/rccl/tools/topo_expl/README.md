# RCCL Topology Explorer (topo_expl)

The RCCL Topology Explorer is a tool for analyzing and exploring network topologies for RCCL (ROCm Communication Collectives Library) collective operations. It simulates various hardware configurations and displays the actual algo/proto combo selections that RCCL would make.

## Building

### Prerequisites
- ROCm/HIP development environment
- RCCL source code
- hipify-perl (for source transformation)

### Build Instructions

```bash
cd tools/topo_expl
make
```

## Usage

```bash
./topo_expl [-m|--model model_id] [-n|--nodes numNodes=1] [-a|--arch gpu_arch] [-t] [-h]
```

### Parameters

- `-m, --model id`: Specifies the topology model to use (required to run a model)
- `-n, --nodes N`: Number of nodes to simulate (default: 1)
- `-a, --arch arch`: Show available models by GPU architecture (e.g., gfx906, gfx908, gfx910, gfx942, gfx950)
- `-t, --test`: Run comprehensive test suite (all models with 1,2,4,8,16 nodes by default)
- `-h, --help`: Display help message and available models

### Test Suite Options

When using `-t` or `--test`, you can further customize the test run:

- `--include-models=ids`: Comma-separated model IDs to test (e.g., `--include-models=0,1,2,4,8`)
- `--include-nodes=counts`: Comma-separated node counts to test (e.g., `--include-nodes=1,4,8`)
- `--exclude-models=ids`: Comma-separated model IDs to exclude from test (e.g., `--exclude-models=57,59`)
- `--exclude-nodes=counts`: Comma-separated node counts to exclude from test (e.g., `--exclude-nodes=16`)

### Available Models

Run `./topo_expl` without arguments to see the list of available models. Each model represents a different hardware configuration. [Each model file](./models) pertains to a particular GPU model and node configuration. It can be output by RCCL through setting the environment variable `NCCL_TOPO_DUMP_FILE`. Model XMLs have been generated for simplicity.

## Example Usage: Print RCCL's algorithm/protocol selections

The tool is typically run with the `NCCL_DEBUG=INFO` environment variable to show the topology information and print out the constructed rings/trees. However, for the convenience of just printing the algo/proto table, we use version `NCCL_DEBUG=version` in this example to avoid printing topo details.

```bash
# List available models
./topo_expl

# List models for a specific GPU architecture
./topo_expl -a gfx942
# Or: ./topo_expl --arch gfx942

# Display help message
./topo_expl -h

# Test MI300 configuration (model 55)
NCCL_DEBUG=version ./topo_expl -m 55

# Test a multi-node MI300 configuration with 8 nodes
NCCL_DEBUG=version ./topo_expl -m 55 -n 8
# Or: ./topo_expl --model 55 --nodes 8

# Test a multi-node MI350 configuration with 2 nodes
NCCL_DEBUG=version ./topo_expl -m 59 -n 2

# Test MI250 configuration (model 42)
NCCL_DEBUG=version ./topo_expl -m 42

# Test a multi-node MI250 configuration with 4 nodes
NCCL_DEBUG=version ./topo_expl -m 42 -n 4
```

## Test Suite Usage

The integrated test suite allows comprehensive testing of all models with various node configurations:

```bash
# Run test suite for all models with default node counts (1,2,4,8,16)
./topo_expl -t

# Test only specific models
./topo_expl -t --include-models=0,1,2,4,8

# Test all models with specific node counts
./topo_expl -t --include-nodes=1,4

# Test all models except specific ones
./topo_expl -t --exclude-models=57,59

# Test all models except 16-node configurations
./topo_expl -t --exclude-nodes=16

# Combine include and exclude options
./topo_expl -t --include-models=0,1,2 --include-nodes=1,4 --exclude-models=1
```


## Sample output

```bash
# cmd used (-m/-n or --model/--nodes)
NCCL_DEBUG=version ./topo_expl -m 55 -n 8
```

```bash

Running fp32 production choices for algorithm/protocol/maxChannels
| Max Size(B)     | Count           | Collective      | Algorithm  | Protocol   | Max Channels |
|-----------------|-----------------|-----------------|------------|------------|--------------|
| 32              | 8               | AllReduce       | Tree       | LL         | 1            |
| 64              | 16              | AllReduce       | Tree       | LL         | 1            |
| 128             | 32              | AllReduce       | Tree       | LL         | 1            |
| 256             | 64              | AllReduce       | Tree       | LL         | 1            |
| 512             | 128             | AllReduce       | Tree       | LL         | 1            |
| 1024            | 256             | AllReduce       | Tree       | LL         | 1            |
| 2048            | 512             | AllReduce       | Tree       | LL         | 1            |
| 4096            | 1024            | AllReduce       | Tree       | LL         | 2            |
| 8192            | 2048            | AllReduce       | Tree       | LL         | 4            |
| 16384           | 4096            | AllReduce       | Tree       | LL         | 8            |
| 32768           | 8192            | AllReduce       | Tree       | LL         | 16           |
| 65536           | 16384           | AllReduce       | Tree       | LL         | 32           |
| 131072          | 32768           | AllReduce       | Tree       | LL         | 64           |
| 262144          | 65536           | AllReduce       | Tree       | LL         | 64           |
| 524288          | 131072          | AllReduce       | Tree       | LL         | 64           |
| 1048576         | 262144          | AllReduce       | Tree       | LL         | 64           |
| 2097152         | 524288          | AllReduce       | Tree       | LL128      | 64           |
| 4194304         | 1048576         | AllReduce       | Tree       | LL128      | 64           |
| 8388608         | 2097152         | AllReduce       | Tree       | LL128      | 64           |
| 16777216        | 4194304         | AllReduce       | Tree       | LL128      | 64           |
| 33554432        | 8388608         | AllReduce       | Tree       | LL128      | 64           |
| 67108864        | 16777216        | AllReduce       | Tree       | Simple     | 64           |
| 134217728       | 33554432        | AllReduce       | Tree       | Simple     | 64           |
| 268435456       | 67108864        | AllReduce       | Tree       | Simple     | 64           |
| 536870912       | 134217728       | AllReduce       | Ring       | Simple     | 64           |
| 1073741824      | 268435456       | AllReduce       | Ring       | Simple     | 64           |
| 2147483648      | 536870912       | AllReduce       | Ring       | Simple     | 64           |
| 4294967296      | 1073741824      | AllReduce       | Ring       | Simple     | 64           |
...
```

