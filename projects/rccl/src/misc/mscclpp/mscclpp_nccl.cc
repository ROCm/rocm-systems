/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt and NOTICES.txt for license information
 ************************************************************************/

#include "mscclpp/mscclpp_nccl.h"

std::unordered_map<ncclUniqueId, mscclppUniqueId> mscclpp_uniqueIdMap;
std::unordered_map<mscclppUniqueId, std::unordered_set<ncclUniqueId>> mscclpp_uniqueIdReverseMap;
std::unordered_map<mscclppComm_t, mscclppUniqueId> mscclpp_commToUniqueIdMap;
std::unordered_map<ncclComm_t, ncclUniqueId> ncclCommToUniqueIdMap;
