// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/cid_manager.hpp"

#include <algorithm>

namespace rocprofiler
{
namespace pc_sampling
{
void
PCSCIDManager::cid_async_activity_completed(context::correlation_id* cid)
{
    // Hold the lock while updating the state of PCSCIDManager
    std::unique_lock<std::mutex> lock(m);
    // The kernel of the `cid` completed, so add cid to `q1`.
    q1.emplace_back(cid);
}

void
PCSCIDManager::manage_cids_implicit(const pc_samples_copy_fn_t& pc_samples_copy_fn)
{
    std::vector<context::correlation_id*> q3;
    {
        // To manipulate the contents of q1 and q2 and change the state of PCSCIDManager,
        // acquire the lock.
        std::unique_lock<std::mutex> lock(m);
        // Move all CIDs from q2 to the q3 local for this function.
        // Note: two buffer flushes happened since kernels of q3's CIDs completed.
        q3 = std::move(q2);
        // Move all CIDs from q1 to q2.
        // Note: exactly one buffer flush occured since kernels of q2's CIDs completed.
        q2 = std::move(q1);

        // We move CIDs from one queue to another to reflect that an implicit ROCr's buffer flush
        // occured. move from q1 to q2 reflects the first buffer flush since kernels of q1's CIDs
        // completed move from q2 to local q3 reflects the second buffer flush since kernels of q2's
        // CIDs completed.

        // Empty the q1 to indicate that there are no CIDs with the following property:
        // no buffer flush occured since the kernel of CID is marked completed.
        q1.clear();

        // The code that follows does not change the state of the PCSCIDManager, so release the lock
        // implicitly.
    }

    // Copy PC samples from the ROCr's buffer to the SDK's buffer by invoking the passed function.
    pc_samples_copy_fn();

    // Exactly two implicit buffer flushes occured since kernels of q3's CIDs completed.
    // Since all PC samples corresponding to these CIDs are placed in the SDK's buffer,
    // decrement their reference counters to indicate that PC sampling service will not use
    // these CIDs anymore.
    // Eventually, CIDs retirement service will report retirement of these CIDs
    // to the client tool.
    // Note: the q3 is local to the function, so there is no need for inter-thread synchronization.
    retire_cids_of(q3);
}

void
PCSCIDManager::manage_cids_explicit(const pc_samples_copy_fn_t& pc_samples_explicit_flush_fn)
{
    std::vector<context::correlation_id*> q1_copy;
    std::vector<context::correlation_id*> q2_copy;
    {
        // To manipulate the contents of q1 and q2 and change the state of PCSCIDManager,
        // acquire the lock.
        std::unique_lock<std::mutex> lock(m);

        // Move all CIDs from q1 and q2 to local q1_copy and q2_copy, respectively
        q1_copy = std::move(q1);
        q2_copy = std::move(q2);

        // Drop CIDs from q1 and q2, because the following explicit flush
        // will deliver corresponding samples.
        q1.clear();
        q2.clear();

        // The code that follows does not change the state of the PCSCIDManager, so release the lock
        // implicitly.
    }

    // Call the passed lambda function to initiate an explicit flush of ROCr buffer by leveraging
    // the `hsa_ven_amd_pcs_flush flush`. The latter function guarantees delivery of all samples
    // generated (sequenced) before the call to the `hsa_ven_amd_pcs_flush`.
    // Thus, all samples corresponding to CIDs of `q1_copy` and `q2_copy` will be copied
    // from the ROCr's buffer to the SDK's buffer,
    // meaning CIDs of `q1_copy` and `q2_copy` will not be used anymore by the PC sampling service.
    pc_samples_explicit_flush_fn();

    // The PC sampling service will not use q1_copy's and q2_copy's CIDs anymore, so it decrements
    // their CIDs. Eventually, CIDs retirement service will report retirement of these CIDs to the
    // client tool. Note: both `q1_copy` and `q2_copy` are local to the function, so there is no
    // need for inter-thread synchronization.
    retire_cids_of(q1_copy);
    retire_cids_of(q2_copy);
}

/**
 * @brief A helper function used to notify that the correlation IDs of @p q
 * are ready to be retired by decrementing their ref_counters.
 * Furthermore, this function notifies the PC sampling parser that
 * kernels matching these CIDs are completed and can be removed from parser's
 * internal maps.
 */
void
PCSCIDManager::retire_cids_of(std::vector<context::correlation_id*>& q)
{
    // This function does not change the local state of the manager,
    // so it does not need synchronization.
    for(auto* cid : q)
    {
        // Notify the parser that the kernel has completed.
        pcs_parser->completeDispatch(cid->internal);
        // Decrement the ref_counter. Eventually, the CID is retired.
        cid->sub_ref_count();
    }
}

}  // namespace pc_sampling
}  // namespace rocprofiler
