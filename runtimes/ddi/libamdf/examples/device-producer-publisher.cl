/* One work-item publishes a prepared packet into a device-mapped queue.
 * AQL writes its header last. The release fences order ring, index, doorbell,
 * and completion stores across the GPU producer and target queue. */
typedef unsigned int uint;
typedef unsigned long ulong;

kernel void amdf_publish_queue(global const uint *commands,
                               global volatile uint *ring,
                               global volatile ulong *write_index,
                               global volatile ulong *doorbell,
                               global volatile uint *completion, uint words,
                               uint header_last, ulong next_index,
                               ulong doorbell_value) {
  if (get_global_id(0) != 0)
    return;
  uint first = header_last ? 1u : 0u;
  for (uint index = first; index < words; ++index)
    ring[index] = commands[index];
  if (header_last) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    ring[0] = commands[0];
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  *write_index = next_index;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  *doorbell = doorbell_value;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  *completion = 1;
}
