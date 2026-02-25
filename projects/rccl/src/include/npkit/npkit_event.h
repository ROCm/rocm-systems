/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef NPKIT_EVENT_H_
#define NPKIT_EVENT_H_

#define NPKIT_EVENT_INVALID                                     0x0

#define NPKIT_EVENT_ALL_REDUCE_RING_ENTRY                       0x1
#define NPKIT_EVENT_ALL_REDUCE_RING_EXIT                        0x2
#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY                0x3
#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT                 0x4
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY                 0x5
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT                  0x6

#define NPKIT_EVENT_COPY_SEND_ENTRY                             0x7
#define NPKIT_EVENT_COPY_SEND_EXIT                              0x8
#define NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY                      0x9
#define NPKIT_EVENT_DIRECT_COPY_SEND_EXIT                       0xA
#define NPKIT_EVENT_DIRECT_RECV_ENTRY                           0xB
#define NPKIT_EVENT_DIRECT_RECV_EXIT                            0xC
#define NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY                 0xD
#define NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT                  0xE
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY          0xF
#define NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT           0x10
#define NPKIT_EVENT_DIRECT_SEND_ENTRY                           0x11
#define NPKIT_EVENT_DIRECT_SEND_EXIT                            0x12
#define NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY               0x13
#define NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT                0x14
#define NPKIT_EVENT_RECV_ENTRY                                  0x15
#define NPKIT_EVENT_RECV_EXIT                                   0x16
#define NPKIT_EVENT_RECV_COPY_SEND_ENTRY                        0x17
#define NPKIT_EVENT_RECV_COPY_SEND_EXIT                         0x18
#define NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY                      0x19
#define NPKIT_EVENT_RECV_REDUCE_COPY_EXIT                       0x1A
#define NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY                 0x1B
#define NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT                  0x1C
#define NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY                      0x1D
#define NPKIT_EVENT_RECV_REDUCE_SEND_EXIT                       0x1E
#define NPKIT_EVENT_SEND_ENTRY                                  0x1F
#define NPKIT_EVENT_SEND_EXIT                                   0x20
#define NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY                      0x21
#define NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT                       0x22

#define NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_ENTRY                 0x23
#define NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_EXIT                  0x24
#define NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY      0x25
#define NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT       0x26

#define NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY                     0x27
#define NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT                      0x28
#define NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY                  0x29
#define NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT                   0x2A

#define NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY                  0x2B
#define NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT                   0x2C
#define NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY               0x2D
#define NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT                0x2E

#define NPKIT_EVENT_NET_SEND_ENTRY                              0x2F
#define NPKIT_EVENT_NET_SEND_EXIT                               0x30

#define NPKIT_EVENT_NET_RECV_ENTRY                              0x31
#define NPKIT_EVENT_NET_RECV_EXIT                               0x32

#define NPKIT_EVENT_TIME_SYNC_GPU                               0x33
#define NPKIT_EVENT_TIME_SYNC_CPU                               0x34

#define NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY                  0x35
#define NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT                   0x36
#define NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY      0x37
#define NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT       0x38
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY  0x39
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT   0x3A
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY 0x3B
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT  0x3C
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY           0x3D
#define NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT            0x3E

#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY         0x3F
#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT          0x40
#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY      0x41
#define NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT       0x42

#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY    0x43
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT     0x44
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY          0x45
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT           0x46
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY       0x47
#define NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT        0x48

#define NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY                  0x49
#define NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT                   0x4A
#define NPKIT_EVENT_SEND_RECV_SEND_ENTRY                        0x4B
#define NPKIT_EVENT_SEND_RECV_SEND_EXIT                         0x4C
#define NPKIT_EVENT_SEND_RECV_RECV_ENTRY                        0x4D
#define NPKIT_EVENT_SEND_RECV_RECV_EXIT                         0x4E

#define NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME                    0x4F

#define NPKIT_EVENT_ALL_GATHER_RING_ENTRY                       0x50
#define NPKIT_EVENT_ALL_GATHER_RING_EXIT                        0x51
#define NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY                  0x52
#define NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT                   0x53
#define NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY        0x54
#define NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT         0x55
#define NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY           0x56
#define NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT            0x57

#define NPKIT_EVENT_NET_TEST_ENTRY                              0x58
#define NPKIT_EVENT_NET_TEST_EXIT                               0x59

#define NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY                      0x5A
#define NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT                       0x5B
#define NPKIT_EVENT_MSCCL_REDUCE_ENTRY                          0x5C
#define NPKIT_EVENT_MSCCL_REDUCE_EXIT                           0x5D
#define NPKIT_EVENT_MSCCL_SEND_ENTRY                          	0x5E
#define NPKIT_EVENT_MSCCL_SEND_EXIT                           	0x5F
#define NPKIT_EVENT_MSCCL_RECV_ENTRY                            0x60
#define NPKIT_EVENT_MSCCL_RECV_EXIT                             0x61
#define NPKIT_EVENT_MSCCL_RUN_ENTRY                             0x62
#define NPKIT_EVENT_MSCCL_RUN_EXIT                              0x63
#define NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY                0x64
#define NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT                 0x65
#define NPKIT_EVENT_MSCCL_INIT_ENTRY                            0x66
#define NPKIT_EVENT_MSCCL_INIT_EXIT                             0x67

#define NPKIT_EVENT_BROADCAST_RING_ENTRY                        0x68
#define NPKIT_EVENT_BROADCAST_RING_EXIT                         0x69

#define NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY                   0x70
#define NPKIT_EVENT_REDUCE_SCATTER_RING_EXIT                    0x71
#define NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_ENTRY              0x72
#define NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_EXIT               0x73
#define NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_ENTRY  0x74
#define NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_EXIT   0x75
#define NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_ENTRY  0x76
#define NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_EXIT   0x77
#endif
