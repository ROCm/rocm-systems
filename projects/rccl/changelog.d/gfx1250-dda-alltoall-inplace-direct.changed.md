* On gfx1250, in-place AllToAll (`sendbuff == recvbuff`) no longer takes the DDA fabric tiers (LL, LL128 or VMM); it takes per-peer Send/Recv at every size. Out-of-place AllToAll is unchanged.
