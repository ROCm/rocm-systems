// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

/* define GDA_QUEUEPAIR_MOCK prior to including queue_pair_provider.hpp
 * to use mock QueuePair objects */

#if   defined(GDA_QUEUEPAIR_MOCK)
#include "queue_pair/queue_pair_mock.hpp"
namespace rocshmem { using QueuePair = QueuePairMock; }
#elif defined(GDA_MUX)
#include "queue_pair_mux.hpp"
namespace rocshmem { using QueuePair = QueuePairMux; }
#elif defined(GDA_IONIC)
#include "gda/ionic/queue_pair_ionic.hpp"
namespace rocshmem { using QueuePair = QueuePairIONIC; }
#elif defined(GDA_BNXT)
#include "gda/bnxt/queue_pair_bnxt.hpp"
namespace rocshmem { using QueuePair = QueuePairBNXT; }
#elif defined(GDA_MLX5)
#include "gda/mlx5/queue_pair_mlx5.hpp"
namespace rocshmem { using QueuePair = QueuePairMLX5; }
#endif

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_PROVIDER_HPP_
