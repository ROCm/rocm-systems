/* Minimal AQL target proving that a GPU-published dispatch executed. */
typedef unsigned int uint;
kernel void amdf_device_target(global volatile uint *completion) {
  if (get_global_id(0) != 0)
    return;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  *completion = 2;
}
