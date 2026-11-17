# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Keep in sync with its copies under projects/rocprofiler-compute/:
#   skills/rocprof-compute-torch-trace/evals/samples/jax_workload.py

"""
Sample JAX workload for rocprof-compute --trace-ml-api jax.

    rocprof-compute profile --experimental --trace-ml-api jax --no-roof \
        -n jax_workload -- python3 ./jax_workload.py
"""

from functools import partial

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
train_step(w, x)

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
mapped.lower(xs)
mapped(xs)


class PmapWorker:
    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def step(self, v):
        return v * v


PmapWorker().step(xs)

mesh = Mesh(jax.devices(), ("x",))
sharded_x = jax.device_put(x, NamedSharding(mesh, PartitionSpec("x", None)))


@jax.jit
def shard_map_square(v):
    return jax.shard_map(
        lambda operand: operand * operand,
        mesh=mesh,
        in_specs=PartitionSpec("x", None),
        out_specs=PartitionSpec("x", None),
    )(v)


shard_map_square(sharded_x)


@jax.shard_map(
    mesh=mesh,
    in_specs=PartitionSpec("x", None),
    out_specs=PartitionSpec("x", None),
)
def shard_map_cube(v):
    return v * v * v


shard_map_cube(sharded_x)


@jax.jit
@jax.shard_map(
    mesh=mesh,
    in_specs=PartitionSpec("x", None),
    out_specs=PartitionSpec("x", None),
)
def shard_map_inc(v):
    return v + 1


shard_map_inc(sharded_x)
