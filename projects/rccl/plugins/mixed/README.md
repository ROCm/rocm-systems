# RCCL Mixed Plugin

The mixed plugin demonstrates that a **single shared object can export more than one
RCCL plugin type**. The example combines the network (net) and tuner plugin APIs in one
library, but the same approach applies to other plugin types (for example, the profiler).

This is useful when a vendor ships a network stack together with a matching tuner or
profiler and wants to distribute and version them as one artifact instead of several
separate `.so` files.

## What the example contains

`example/plugin.c` exports two versioned plugin symbols from one source file:

- `ncclNetPlugin_v12` &mdash; the network plugin.
- `ncclTunerPlugin_v6` &mdash; the tuner plugin.

## Building

```shell
cd example
make            # produces libnccl-mixed.so
make test       # builds, then checks both plugin symbols are exported
```

You can also inspect the exported symbols directly:

```shell
nm -D libnccl-mixed.so | grep -E "NetPlugin|TunerPlugin"
```

## Using it at runtime

Point RCCL at the mixed library through the network plugin variable, and leave the
dedicated tuner/profiler variables unset so RCCL reuses the same object for them:

```shell
export NCCL_NET_PLUGIN=/path/to/libnccl-mixed.so
unset NCCL_TUNER_PLUGIN
unset NCCL_PROFILER_PLUGIN
```

With `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,TUNING`, the log shows both the
net plugin and the tuner plugin being loaded from the same object:

```text
NET/Plugin: Loaded net plugin Plugin (v12)
Successfully loaded external network plugin /path/to/libnccl-mixed.so
TUNER/Plugin: Using Plugin (v6)
```

The example net plugin advertises zero devices, so a `Failed to initialize NET plugin`
message is expected; it does not affect the tuner reuse being demonstrated.

## Further reading

- Automated tests: `test/ext-plugins/tests/ext-mixed/test_mixed_plugin.py`
