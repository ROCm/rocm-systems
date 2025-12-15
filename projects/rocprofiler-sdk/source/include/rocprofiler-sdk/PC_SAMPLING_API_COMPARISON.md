# PC Sampling API: Flags vs Multiple Calls Comparison

## Summary

**Flags Approach:** Users specify the record version and use flags to control which record types to receive (valid only, valid+invalid, etc.) in a single configuration call. Example: `configure(..., VERSION_2, sizeof(v2), INCLUDE_INVALID_SAMPLES)`.

**Multiple Calls Approach:** Users make multiple configuration calls - first call locks the valid record version, subsequent calls with VERSION_0 enable invalid sample delivery. The order matters: valid version must be locked before enabling v0. Example: call with VERSION_2 to lock format, then call again with VERSION_0 to enable invalid samples.

## Comparison Table

| Aspect | Flags | Multiple Calls |
|--------|-------|----------------|
| **Calls needed** | 1 | 1-2 |
| **Complexity** | Simple, single call | Complex ordering rules |
| **Error-prone** | No | Yes (wrong order, v0-only ambiguity) |
| **Mental model** | "What do I want to receive?" | "Lock format, then add options" |
| **API additions** | +1 flags enum | None |
| **Documentation** | 2-3 examples | 5+ examples + locking semantics |
| **Similar pattern** | Standard flags pattern | Buffer tracing (but different use case) |
| **Extensibility** | Add new flags easily | More complex state tracking |
| **User clarity** | Self-documenting flag names | "Why configure twice?" |

## Pros & Cons

### Flags Approach
**Pros:**
- Single call does everything - simple and clear
- Hard to misuse - no ordering requirements
- Self-documenting - `INCLUDE_INVALID_SAMPLES` is obvious
- Standard pattern familiar to users
- Easy to extend with new flags

**Cons:**
- Adds one new enum type to API surface

### Multiple Calls Approach
**Pros:**
- Similar to `rocprofiler_configure_buffer_tracing_service` pattern
- No new types needed

**Cons:**
- Requires two calls for common use case (valid + invalid)
- Complex locking semantics: "lock format first, then enable v0"
- Easy to misuse: wrong order or v0-only causes errors
- Documentation burden: needs extensive explanation
- Confusing: "Why call configure twice for one service?"
- Not truly analogous to buffer tracing (that enables different *services*, this controls *delivery options* for same service)

## Recommendation

**Use Flags Approach.** It's simpler, clearer, and follows the principle that configuration options for a single service should be flags, not multiple function calls. The multiple-calls pattern works well when enabling different services (HIP tracing, HSA tracing, etc.), but PC sampling record delivery is just a configuration choice for one service.
