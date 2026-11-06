# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Keep in sync with the original under projects/rocprofiler-compute/:
#   sample/jax_workload.py

"""
Sample JAX workload for rocprof-compute --trace-ml-api jax.

    rocprof-compute profile --experimental --trace-ml-api jax --no-roof \
        -n jax_workload -- python3 ./jax_workload.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@jax.jit
def matmul_relu(a, b):
    return jnp.maximum(a @ b, 0.0)


@jax.jit
def train_step(w, x):
    y = matmul_relu(x, w)
    return jnp.sum(y * y)


w = jnp.ones((512, 512))
x = jnp.ones((512, 512))
train_step(w, x)  # line 13

matmul_relu(w, x)
x @ x

device_count = len(jax.devices())
xs = jnp.stack([x] * device_count)


@jax.pmap
def pmap_square(v):
    return v * v


pmap_square(xs)


def pmap_double(v):
    return v + v


mapped = jax.pmap(pmap_double)
mapped(xs)


@jax.jit
def shard_map_square(v):
    return jax.shard_map(
        lambda operand: operand * operand,
        mesh=Mesh(jax.devices(), ("x",)),
        in_specs=PartitionSpec("x", None),
        out_specs=PartitionSpec("x", None),
    )(v)


mesh = Mesh(jax.devices(), ("x",))
sharded_x = jax.device_put(x, NamedSharding(mesh, PartitionSpec("x", None)))
shard_map_square(sharded_x)
