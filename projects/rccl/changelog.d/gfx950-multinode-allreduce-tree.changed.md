* On MI350X (gfx950), multi-node AllReduce runs Tree instead of Ring up to 4 MiB per rank (32 MiB per node with 8 GPUs per node). Tree/LL128 is 12-32% faster than Ring/LL128 there on 4 and 8 nodes.
