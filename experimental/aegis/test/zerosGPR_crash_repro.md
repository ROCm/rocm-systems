# AegisBit Heap Corruption — Root Cause Analysis

## Summary

82 of 100 randomly-sampled `FLASH_ATTN_EXT` tests crashed when instrumented with AegisBit (pre-fix). All 100 pass without AegisBit. The failures were host-side heap corruption during process cleanup, affecting all `hsk` values.

**Status: FIXED.** 50/50 runs pass after the fix. 20/20 diverse hsk/kv combinations pass with zero crashes and zero numerical errors.

## Root Cause

**The ELF patching pipeline is clean.** ASAN unit tests confirm zero memory safety violations in the patching code.

**The crash was caused by nested HSA InterceptQueue wrappers.** AegisBit's `LD_PRELOAD` interposition of `hsa_queue_create` called `hsa_amd_queue_intercept_create` to create an interceptable queue. But rocprofiler-sdk (which loads AegisBit as a tool) ALSO replaces `hsa_queue_create` in the HSA CoreApiTable with its own version that calls `hsa_amd_queue_intercept_create`.

The result was a double-wrapped queue:
```
InterceptQueue(AegisBit) → InterceptQueue(rocprofiler-sdk) → RealQueue
```

During `HSA::Runtime::Unload()`, the nested `InterceptQueue` wrappers — each with their own heap-allocated bookkeeping — were torn down in an order that corrupted glibc heap metadata. This manifested as `corrupted size vs. prev_size`, `corrupted double-linked list`, and segfaults during process exit.

### Evidence

1. `AEGISBIT_ENABLED=0` (bypasses queue interception): 25/25 pass
2. `AEGISBIT_SKIP_KERNEL='.*'` (no patching, but queue interception active): 15/25 fail
3. Minimal rocprofiler-sdk tool (no queue interception): 25/25 pass
4. Without `LD_PRELOAD`: 25/25 pass

## Fix

Two changes:

### 1. Eliminate nested InterceptQueue wrappers (`HSAInterceptor.cpp`)

Both the `LD_PRELOAD` `hsa_queue_create` interposition and the CoreApiTable `interceptedQueueCreate` now call the REAL `hsa_queue_create` (via `dlsym(RTLD_NEXT)` / `S.OriginalQueueCreateFn`), which rocprofiler-sdk already wraps to create a single-layer `InterceptQueue`. AegisBit then registers its packet handler on the resulting queue via `hsa_amd_queue_intercept_register`, avoiding the second wrapper.

### 2. Skip HSA calls during finalize (`TracingEngine.cpp`)

`cleanupDispatch()` now accepts a `DuringFinalize` parameter. When called from `finalize()`, it skips `hsa_signal_destroy`, `hsa_signal_store_relaxed`, and `releaseGpuKernarg` — the HSA runtime may be partially torn down at this point.

### 3. Restore original `hsa_queue_create` during uninstall (`HSAInterceptor.cpp`)

`HSAInterceptor::uninstall()` now restores the original `hsa_queue_create` function pointer in the CoreApiTable before clearing state.

## Reproduction (pre-fix)

```bash
TESTBIN=/home/djavady/llama_cpp/build-hip/bin/test-backend-ops
LIB=/home/djavady/aegis_two/build/src/libaegisbit.so

# Single test (crashed ~80% of the time before fix):
LD_PRELOAD=$LIB $TESTBIN -o FLASH_ATTN_EXT -b ROCm0 \
    -p "hsk=64,hsv=64,nh=4,nr23=\[1,1\],kv=113,nb=1,mask=1,sinks=1,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_KV=f16,permute=\[0,1,2,3\]"
```

## Pre-fix Results (100 random tests)

**82 fail, 18 pass.**

| Reason | Count |
|--------|-------|
| `corrupted size vs. prev_size` | 32 |
| `corrupted double-linked list` | 26 |
| segfault (no message) | 16 |
| `corrupted size vs. prev_size in fastbins` | 5 |
| other | 3 |

## Post-fix Results

- 50/50 single-test runs: **all pass**
- 20/20 diverse hsk/kv combinations: **all pass, zero numerical errors**

## Unit Test

`test/unit/HeapCorruptionReproGTest.cpp` patches a captured FLASH_ATTN_EXT code object (`test/unit/fixtures/flash_attn_gfx950.bin`) under ASAN. Confirms no buffer overflow in the patching pipeline.
