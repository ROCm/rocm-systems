# RCCL Tuner Plugin API Overview

This document describes the API structure to be implemented by an external tuner plugin for RCCL. The purpose of this plugin is to enable stakeholders to hand-tailor the selection of an algorithm, a protocol, number of channels (thread blocks) based on an input configuration of interest: message size, number of nodes and GPUs, and link types (PCIe, XGMI, NET).

## Notes
- The [model demo plugin](model_demo/plugin.c) is only a demonstration that uses math models to approximate BW and latency of available choices of algorithms and protocols and provide the one that scores the lowest latency. It is customized for MI300 GPUs and RoCEv2 networks on a limited number of nodes. It is not meant to be inclusive of all AMD GPUs/Network setups out there. 
- The API allows partial outputs: tuners can set only the algorithm and protocol, or let RCCL set the remaining fields (e.g., number of channels).
- If`getCollInfo()`fails, RCCL will use its default internal mechanisms to determine the best collective configuration.
- `getCollInfo()`is called for each collective invocation per communicator, so special care is to be taken not to cause excessive latency.
- The advantage of this plugin is that each customer can create and maintain their hand-tailored tuner without relying on RCCL to develop and maintain it.
- Supported RCCL algorithms are `NCCL_ALGO_TREE` and `NCCL_ALGO_RING`.
- Supported RCCL protocols are `NCCL_PROTO_SIMPLE`, `NCCL_PROTO_LL` and `NCCL_PROTO_LL128`.
  - Until support is present for network collectives, we show in our example how to ignore other algorithms in `pluginGetCollInfo` API implementation as follows:
    ```C++
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1) continue;
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1) continue;
    ```
---
# API Description 
The `ncclTuner_v1_t` structure must be implemented to build a custom tuner. 

## Structure: `ncclTuner_v1_t`

### Fields

#### 1. `name`
  Type: `const char*`  
  Description: The name of the tuner. Can be used for logging purposes when (`NCCL_DEBUG=info NCCL_DEBUG_SUBSYS=tune`) are set.

### Functions

#### 1. `init` (called upon communicator initialization with `ncclCommInitRank`)

Initializes the tuner states. Each communicator initializes its tuner. nNodes x nRanks = total number of GPUs participating in the collective communication

- **Parameters**:
  - `nRanks` (size_t): The number of devices (GPUs).
  - `nNodes` (size_t): The number of OS nodes (physical nodes or VMs).
  - `logFunction` (ncclDebugLogger_t): A log function that can be useful to turn on certain debugging info.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the initialization.

#### 2. `getCollInfo` (called for each collective call per communicator)

Retrieves information about the collective algorithm, protocol, and number of channels for the given input parameters.

- **Parameters**:
  - `collType` (ncclFunc_t): The collective type, e.g., `allreduce`, `allgather`, etc.
  - `nBytes` (size_t): The size of the collective in bytes.
  - `collNetSupport` (int): Whether `collNet` supports this type.
  - `nvlsSupport` (int): Whether NVLink SHARP supports this type.
  - `numPipeOps` (int): The number of operations in the group.
  
- **Outputs**:
  - `algorithm` (int*): The selected algorithm to be used for the given collective.
  - `protocol` (int*): The selected protocol to be used for the given collective.
  - `nChannels` (int*): The number of channels (and SMs) to be used.
  
- **Description**:
  If `getCollInfo()` does not return `ncclSuccess`, RCCL will fall back to its default tuning for the given collective. The tuner is allowed to leave fields unset, in which case RCCL will automatically set those fields.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the operation.

#### 3. `destroy` (called upon communicator finalization with `ncclCommFinalize`)

Terminates the plugin and cleans up any resources allocated by the tuner.

- **Return**:  
  Type: `ncclResult_t`  
  The result of the cleanup process.

---


# Build instructions and usage

- The way to use the external plugin is to implement the desired algorithm/protocol selection technique using the API described above. `ext-tuner/model_demo/plugin.c` is an example based on MI300 tuning table by default as a reference for customers in `plugin.c`.
- Build the `libnccl-tuner.so` file following [the Makefile example](model_demo/Makefile). 

## Building and using example libnccl-tuner.so
```bash
cd $RCCL_HOME/ext-tuner/model_demo/ 
make
```
Next is to let RCCL know that you want to use the custom-made libnccl-tuner.so by setting the following environment variable to the directory of the libnccl-tuner.so file:

```bash
export NCCL_TUNER_PLUGIN=$RCCL_HOME/ext-tuner/model_demo/libnccl-tuner.so
```

