* gfx950 single-node 8-rank `ncclBroadcast` from 8 MiB + 128 bytes to 64 MiB takes the pipelined DDA IPC kernel on 56 blocks instead of the ring, when the size is a multiple of 128 bytes.
