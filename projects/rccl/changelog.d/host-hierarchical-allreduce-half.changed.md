* `RCCL_HOST_HIER_ALLREDUCE`: the default (`-1`) now also uses host-hierarchical AllReduce for FP16 and BF16 `ncclSum` on two gfx1250 hosts with eight ranks each, for 2 to 512 MiB.
