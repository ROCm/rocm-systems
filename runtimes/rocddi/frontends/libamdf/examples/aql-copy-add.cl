/* Qualification kernel for the direct AQL example. The host program controls
 * publication and completion; this kernel deliberately performs all work in
 * one work-item so result validation does not depend on scheduler ordering. */
typedef unsigned int uint;

/* One work-item deliberately performs the complete operation. Volatile,
 * dynamically indexed private storage forces a real scratch allocation while
 * keeping result checking independent of workgroup scheduling. */
kernel void amdf_copy_add(global const uint *source,
                          global uint *target,
                          global volatile uint *completion,
                          global volatile uint *gate,
                          global volatile uint *started,
                          uint count,
                          uint addend,
                          uint token) {
  if (started != 0) {
    *started = token;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  }
  if (gate != 0) {
    while (*gate == 0u) {}
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }

  volatile uint scratch[8];
  uint cursor = token & 7u;
  for (uint index = 0; index < 8; ++index) {
    scratch[cursor] = addend;
    cursor = (cursor + 3u) & 7u;
  }

  for (uint index = 0; index < count; ++index)
    target[index] = source[index] + scratch[(index + token) & 7u];

  /* Publish all payload stores to the system before the completion token. */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  *completion = token;
}
