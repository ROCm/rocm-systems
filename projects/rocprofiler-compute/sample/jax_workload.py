# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Sample JAX workload for rocprof-compute --trace-ml-api jax.

    rocprof-compute profile --experimental --trace-ml-api jax --no-roof \
        -n jax_workload -- python3 ./jax_workload.py
"""

import sys

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


@jax.pmap
def pmap_square(v):
    return v * v


def shard_map_cube(v):
    return v * v * v


def run_workload():
    w = jnp.ones((64, 64))
    x = jnp.ones((64, 64))
    train_step(w, x)
    matmul_relu(w, x)
    x @ x

    devices = jax.devices()
    xs = jnp.stack([x] * len(devices))
    pmap_square(xs)

    mesh = Mesh(devices, ("x",))
    sharded_x = jax.device_put(x, NamedSharding(mesh, PartitionSpec("x", None)))
    jax.shard_map(
        mesh=mesh,
        in_specs=PartitionSpec("x", None),
        out_specs=PartitionSpec("x", None),
    )(shard_map_cube)(sharded_x)


def main():
    devices = jax.devices()
    if not any(getattr(device, "platform", "") == "gpu" for device in devices):
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)
    run_workload()
    print("JAX workload completed")


if __name__ == "__main__":
    main()
