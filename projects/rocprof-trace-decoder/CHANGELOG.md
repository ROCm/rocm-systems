# Changelog for release 0.1.7

### Resolved issues

- Fixed fp64 operations reporting incorrect cycles in gfx11/11.5
- Fixed ds_bvh_ instructions in gfx12
- Fixed s_ttracedata and s_wakeup not properly parsing in MI3xx

### Changes

- s_barrier in gfx11/11.5 is now MSG type and also reports wait time as stalled time, instead of idle.
  - Similar to previous gfx9 versions, this can be misleading in MSG bus utilization and tools shall aware of it.
- s_barrier in gfx9 (MI2xx, MI3xx) now reports two tokens:
  - MSG when entered
  - IMMED when exiting
  - This means tools need to be aware they'll get double the hitcount in gfx9 s_barriers
  - The previous behavior could be misleading on messagebus utilization
- s_endpgm now only reports stall and execute time for the message part.
  - Previously, s_endpgm would display a large MSG use.

### Known issues

- The wave completion part of s_endpgm time is now 'idle' time. This is incoherent with s_waitcnt and s_barrier which considers it a wait (stall).
- In gfx10 and gfx11, the latency of s_barrier is ambiguous when there are some types of surrounding IMMEDs.
  - Consider adding s_nop 0 around barriers when accurate timing is required.
