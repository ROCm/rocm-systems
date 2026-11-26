// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA environment policy, independent of passive native discovery.

/// `ROCr` copy policy disables SDMA only for the explicit value `0`.
/// Unset, `1`, and unrecognized values retain native capability selection.
pub(super) fn sdma_enabled(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Counted hardware queue limit per agent and priority, sampled at startup.
/// Zero deliberately prevents acquisition; invalid or absent values use four.
pub(super) fn counted_queue_limit(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(4)
}

/// `ROCr` disables default queue-error reporting and termination only for `0`.
pub(super) fn queue_fault_message_enabled(value: Option<&str>) -> bool {
    value != Some("0")
}

pub(super) mod cu_mask;

/// Selects native GPU ordinals in the order requested by `ROCr` visibility.
/// An invalid, ambiguous or repeated token terminates the accepted prefix.
/// Missing UUIDs can be selected by ordinal but cannot match a UUID token.
pub(super) fn visible_gpu_indices(ids: &[Option<u64>], selection: Option<&str>) -> Vec<usize> {
    let Some(selection) = selection else {
        return (0..ids.len()).collect();
    };
    let mut selected = Vec::new();
    for token in selection.split(',').take(ids.len()) {
        let token = token.trim().to_ascii_uppercase();
        let index = if token.starts_with('G') {
            if !(5..=20).contains(&token.len()) {
                break;
            }
            let mut matches = ids.iter().enumerate().filter_map(|(index, id)| {
                let id = id.filter(|id| *id != 0)?;
                format!("GPU-{id:016X}")
                    .starts_with(&token)
                    .then_some(index)
            });
            let found = matches.next();
            if matches.next().is_some() {
                break;
            }
            found
        } else {
            let digits = token.strip_prefix('+').unwrap_or(&token);
            let (digits, radix) = if let Some(hex) = digits.strip_prefix("0X") {
                (hex, 16)
            } else if digits.len() > 1 && digits.starts_with('0') {
                (&digits[1..], 8)
            } else {
                (digits, 10)
            };
            usize::from_str_radix(digits, radix).ok()
        };
        let Some(index) = index.filter(|index| *index < ids.len()) else {
            break;
        };
        if selected.contains(&index) {
            break;
        }
        selected.push(index);
    }
    selected
}

/// Admits a kernel agent only when all mandatory execution mechanisms exist.
pub(super) fn supports_execution(gpu: &rocddi::topology::GpuInfo) -> bool {
    rocloader::supports_target(rocloader::Target::from(gpu))
        && crate::memory::compute_copy::supports_target(gpu)
        && gpu.queues.aql.is_some_and(|queue| queue.dispatch_scratch)
        && gpu
            .target_profile
            .scratch_encoding(gpu.wavefront_size)
            .is_some()
        && gpu.queues.copy.is_some()
}

/// Immutable frontend policy for one initialized HSA generation.
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent immutable frontend policies"
)]
pub(crate) struct HsaConfig {
    /// Fatal GPU capture and output naming policy sampled at initialization.
    pub(crate) core_dump: crate::core_dump::Config,

    /// Whether an AQL error without an application callback reports and aborts.
    pub(crate) queue_fault_message_enabled: bool,
    /// Whether polling-only scalar waits back off after their hot-spin window.
    pub(crate) yielding_poll_enabled: bool,
    /// Generation-wide copy-engine configuration.
    pub(crate) sdma_enabled: bool,
    /// Generation-wide peer-copy engine configuration.
    pub(crate) peer_sdma_enabled: bool,
    /// Whether ROCr-compatible fragment allocation is disabled by the caller.
    pub(crate) disable_fragment_allocator: bool,
}

#[cfg(test)]
mod tests {
    use super::visible_gpu_indices;

    #[test]
    fn queue_fault_message_requires_exact_zero_to_disable() {
        for value in [None, Some(""), Some("1"), Some("false"), Some(" 0")] {
            assert!(super::queue_fault_message_enabled(value));
        }
        assert!(!super::queue_fault_message_enabled(Some("0")));
    }

    #[test]
    fn counted_queue_limit_preserves_zero_and_uses_default_when_unspecified() {
        for (value, expected) in [
            (None, 4),
            (Some(""), 4),
            (Some("1"), 1),
            (Some("2"), 2),
            (Some("4"), 4),
            (Some("0"), 0),
            (Some(" 2 "), 2),
            (Some("-1"), 4),
            (Some("invalid"), 4),
        ] {
            assert_eq!(super::counted_queue_limit(value), expected);
        }
    }

    #[test]
    fn sdma_override_disables_only_the_explicit_zero_value() {
        assert!(!super::sdma_enabled(Some("0")));
        for value in [
            None,
            Some("1"),
            Some("-1"),
            Some(""),
            Some(" 0"),
            Some("false"),
        ] {
            assert!(super::sdma_enabled(value));
        }
    }

    #[test]
    fn visibility_preserves_order_and_distinguishes_unset_from_empty() {
        let ids = [Some(0xabc0), Some(0xdef0), None];
        assert_eq!(visible_gpu_indices(&ids, None), [0, 1, 2]);
        assert!(visible_gpu_indices(&ids, Some("")).is_empty());
        assert_eq!(visible_gpu_indices(&ids, Some("2,0x1,00")), [2, 1, 0]);
        assert_eq!(
            visible_gpu_indices(&ids, Some("  GPU-000000000000DEF ,0")),
            [1, 0]
        );
    }

    #[test]
    fn visibility_stops_at_invalid_ambiguous_and_repeated_tokens() {
        let ids = [Some(0xabc0), Some(0xabc1), Some(0)];
        for selection in ["1,-1,0", "1,3,0", "1,,0", "1,1,0", "1,GPU-0000,0"] {
            assert_eq!(visible_gpu_indices(&ids, Some(selection)), [1]);
        }
        assert!(visible_gpu_indices(&ids, Some("GPU-0000000000000000")).is_empty());
    }
}
